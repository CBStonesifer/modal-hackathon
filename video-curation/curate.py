"""Tier 1: per-frame quality from Zarr shard indices, without downloading pixels.

    modal run curate.py --limit 20                    # trial
    modal run --detach curate.py --limit 0            # whole slice

Each episode costs one LIST plus one 16 KB ranged GET. The shard index gives the exact
compressed byte length of every frame. Results are small enough to return directly — no Volume.

Naming note: byte size tracks *image detail*, not sharpness. It correlates r=+0.95 with
Laplacian variance, but Laplacian variance is also a detail measure — and the series drifts
smoothly (lag-1 autocorr +0.62) rather than spiking, and low frames cluster 2x in the final 10%
of episodes. That is scene content (hands entering and leaving view), not motion blur. Hence
`low_detail_frac`, not `blur_frac`.
"""

import modal

app = modal.App("egoverse-curator")

image = modal.Image.debian_slim().uv_pip_install("boto3", "numpy")

INDEX_ENTRIES = 1000  # inner chunks per shard, from the array's chunk_grid
INDEX_BYTES = INDEX_ENTRIES * 16 + 4  # (offset, nbytes) u64 pairs + trailing crc32c
EMPTY_CHUNK = 2**64 - 1


def s3_client():
    import os

    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def frame_sizes(client, bucket: str, prefix: str, camera: str) -> list[int]:
    import numpy as np

    listing = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/{camera}/c/")
    shards = sorted(o["Key"] for o in listing.get("Contents", []))
    if not shards:
        raise FileNotFoundError(f"no shards under {prefix}/{camera}/c/")

    sizes: list[int] = []
    for key in shards:
        tail = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=-{INDEX_BYTES}")
        entries = np.frombuffer(tail["Body"].read()[:-4], dtype="<u8").reshape(INDEX_ENTRIES, 2)
        sizes.extend(int(n) for n in entries[entries[:, 1] != EMPTY_CHUNK][:, 1])
    return sizes


def quality(sizes: list[int]) -> dict:
    import numpy as np

    x = np.asarray(sizes, dtype=np.float64)
    spread = x.std() or 1.0
    z = (x - np.median(x)) / spread

    low = z < -1.0
    runs, current = [0], 0
    for flag in low:
        current = current + 1 if flag else 0
        runs.append(current)

    steps = np.diff(x) if len(x) > 1 else np.zeros(1)
    return {
        "n_frames": int(len(x)),
        "low_detail_frac": float((z < -2.0).mean()),
        "faint_frac": float(low.mean()),
        "longest_faint_run": int(max(runs)),
        "size_cv": float(spread / x.mean()),
        "min_z": float(z.min()),
        "max_step_drop": float(steps.min() / spread),
        "median_bytes": float(np.median(x)),
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("egoverse-r2")],
    max_containers=50,
    retries=modal.Retries(max_retries=3),
    timeout=300,
)
def scan_episode(row: dict) -> dict:
    import os

    bucket = os.environ.get("BUCKET", "rldb")
    prefix = row["zarr_processed_path"].replace(f"s3://{bucket}/", "").rstrip("/")
    base = {"episode_hash": row["episode_hash"], "operator": row.get("operator", "")}

    try:
        client = s3_client()
        sizes = frame_sizes(client, bucket, prefix, row.get("camera", "images.front_1"))
    except Exception as exc:
        return {**base, "ok": False, "error": f"{type(exc).__name__}: {exc}", "sizes": []}

    declared = row.get("num_frames")
    return {
        **base,
        "ok": True,
        "error": "",
        # the shard index should agree with SQL; a mismatch means one of them is lying
        "declared_frames": int(declared) if declared and declared > 0 else -1,
        **quality(sizes),
        "sizes": sizes,
    }


# --------------------------------------------------------------------------- tier 2: poses

zarr_image = modal.Image.debian_slim().uv_pip_install("boto3", "numpy", "zarr==3.1.5")


def spectral_arc_length(speed, fs: float, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """SPARC (Balasubramanian et al.) — smoothness, invariant to amplitude and duration."""
    import numpy as np

    if len(speed) < 4 or speed.max() <= 0:
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + 4))
    freq = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()

    band = freq <= fc
    freq, mag = freq[band], mag[band]
    above = np.flatnonzero(mag >= amp_th)
    if len(above) < 2:
        return 0.0
    freq, mag = freq[above[0] : above[-1] + 1], mag[above[0] : above[-1] + 1]
    df = np.diff(freq) / (freq[-1] - freq[0])
    return float(-np.sum(np.sqrt(df**2 + np.diff(mag) ** 2)))


def motion_features(pose, fps: float, prefix: str) -> dict:
    import numpy as np

    xyz = pose[:, :3]
    dt = 1.0 / fps
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    speed = step / dt
    accel = np.diff(speed) / dt
    jerk = np.diff(accel) / dt

    # quaternion sign is ambiguous, so use the absolute inner product for angular distance
    q = pose[:, 3:]
    dot = np.abs((q[:-1] * q[1:]).sum(1)).clip(0, 1)
    ang = 2 * np.arccos(dot)

    # threshold relative to this episode's own peak, not a percentile — a percentile threshold
    # is degenerate (75% of frames are above p25 by construction, so the span is always ~[0,1])
    active = np.flatnonzero(speed > 0.15 * speed.max())
    return {
        f"{prefix}_path_len": float(step.sum()),
        f"{prefix}_speed_max": float(speed.max()),
        f"{prefix}_speed_mean": float(speed.mean()),
        f"{prefix}_jerk_rms": float(np.sqrt((jerk**2).mean())) if len(jerk) else 0.0,
        f"{prefix}_sparc": spectral_arc_length(speed, fps),
        f"{prefix}_idle_frac": float((speed < 0.01).mean()),
        f"{prefix}_ang_path": float(ang.sum()),
        f"{prefix}_net_disp": float(np.linalg.norm(xyz[-1] - xyz[0])),
        f"{prefix}_straightness": float(np.linalg.norm(xyz[-1] - xyz[0]) / step.sum()) if step.sum() else 0.0,
        # motion-energy span: the sub-interval that actually contains movement
        f"{prefix}_act_start": float(active[0] / len(speed)) if len(active) else 0.0,
        f"{prefix}_act_end": float(active[-1] / len(speed)) if len(active) else 1.0,
    }


@app.function(
    image=zarr_image,
    secrets=[modal.Secret.from_name("egoverse-r2")],
    max_containers=50,
    retries=modal.Retries(max_retries=3),
    timeout=600,
)
def scan_poses(row: dict) -> dict:
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import zarr

    bucket = os.environ.get("BUCKET", "rldb")
    prefix = row["zarr_processed_path"].replace(f"s3://{bucket}/", "").rstrip("/")
    base = {"episode_hash": row["episode_hash"]}
    work = Path(tempfile.mkdtemp())

    try:
        client = s3_client()
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix) + 1 :]
                if not rel or rel.startswith("images"):
                    continue
                target = work / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, obj["Key"], str(target))

        group = zarr.open_group(str(work), mode="r")
        n = int(group.attrs.get("total_frames") or 0)
        fps = float(group.attrs.get("fps") or 30.0)
        keys = {name for name, _ in group.members(max_depth=None)}

        # mecka has no timestamps, so the declared fps has to be taken on faith
        true_fps = fps
        if "obs_rgb_timestamps_ns" in keys:
            stamps = np.asarray(group["obs_rgb_timestamps_ns"][:n])
            stamps = stamps[stamps > 0]
            if len(stamps) > 1:
                true_fps = len(stamps) / ((stamps[-1] - stamps[0]) / 1e9)

        out = {**base, "ok": True, "error": "", "n_frames": n, "fps_attr": fps,
               "fps_true": true_fps, "duration_s": n / true_fps}

        for side in ["left", "right"]:
            key = f"{side}.obs_ee_pose"
            if key in keys:
                out.update(motion_features(np.asarray(group[key][:n]), true_fps, side))
        if "obs_head_pose" in keys:
            out.update(motion_features(np.asarray(group["obs_head_pose"][:n]), true_fps, "head"))

        for side in ["left", "right"]:
            key = f"{side}.obs_keypoints"
            if key in keys:
                kp = np.asarray(group[key][:n]).reshape(n, 21, 3)
                span = np.linalg.norm(kp.max(1) - kp.min(1), axis=1)
                out[f"{side}_hand_span_mean"] = float(span.mean())
                out[f"{side}_hand_span_cv"] = float(span.std() / span.mean()) if span.mean() else 0.0
        return out
    except Exception as exc:
        return {**base, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.local_entrypoint()
def poses(lab: str = "mecka", task: str = "cup_on_saucer", limit: int = 20,
          index: str = "episodes.parquet", out: str = "kinematics.parquet"):
    import pandas as pd

    catalog = pd.read_parquet(index)
    slice_ = catalog[
        ~catalog.is_deleted
        & catalog.lab.eq(lab)
        & catalog.task.eq(task)
        & catalog.zarr_processed_path.fillna("").str.strip().ne("")
    ]
    if limit:
        slice_ = slice_.sort_values("num_frames").iloc[:: max(1, len(slice_) // limit)].head(limit)
    rows = slice_[["episode_hash", "zarr_processed_path"]].to_dict("records")
    print(f"reading poses for {len(rows)} episodes from {lab}/{task}")

    results = list(scan_poses.map(rows, return_exceptions=True))
    good = [r for r in results if isinstance(r, dict) and r.get("ok")]
    bad = [r for r in results if not (isinstance(r, dict) and r.get("ok"))]
    print(f"\n{len(good)} ok, {len(bad)} failed")
    for r in bad[:5]:
        print("  ", r if isinstance(r, Exception) else f"{r['episode_hash']}: {r['error']}")

    df = pd.DataFrame(good)
    df.to_parquet(out, index=False)
    print(f"\n{len(df)} episodes x {len(df.columns)} features -> {out}")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]].to_string(float_format=lambda x: f"{x:.4f}"))


@app.function(
    image=zarr_image,
    secrets=[modal.Secret.from_name("egoverse-r2")],
    max_containers=50,
    retries=modal.Retries(max_retries=3),
    timeout=600,
)
def fetch_traj(row: dict) -> dict:
    """Raw EE trajectory for the proxy-policy ablation. The poses ARE the actions, so this is
    the ground truth a downstream BC policy would be fit to."""
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import zarr

    bucket = os.environ.get("BUCKET", "rldb")
    prefix = row["zarr_processed_path"].replace(f"s3://{bucket}/", "").rstrip("/")
    work = Path(tempfile.mkdtemp())
    try:
        client = s3_client()
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix) + 1 :]
                if not rel or rel.startswith("images"):
                    continue
                target = work / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, obj["Key"], str(target))

        group = zarr.open_group(str(work), mode="r")
        n = int(group.attrs.get("total_frames") or 0)
        keys = {name for name, _ in group.members(max_depth=None)}
        sides = [f"{s}.obs_ee_pose" for s in ("left", "right") if f"{s}.obs_ee_pose" in keys]
        if not sides or n < 24:
            return {"episode_hash": row["episode_hash"], "ok": False, "error": "no usable poses"}

        traj = np.concatenate([np.asarray(group[k][:n]) for k in sides], axis=1)
        # trim to the active span so the ablation is not dominated by dead air
        lo = int(row.get("act_start", 0.0) * (n - 1))
        hi = max(int(row.get("act_end", 1.0) * (n - 1)), lo + 24)
        traj = traj[lo : hi + 1]
        return {"episode_hash": row["episode_hash"], "ok": True, "error": "",
                "traj": traj.astype(np.float32)}
    except Exception as exc:
        return {"episode_hash": row["episode_hash"], "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.local_entrypoint()
def trajectories(joined: str = "joined.parquet", sample: int = 0, out: str = "traj.npz"):
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(joined)
    if "zarr_processed_path" not in df.columns:
        cat = pd.read_parquet("episodes.parquet")[["episode_hash", "zarr_processed_path"]]
        df = df.merge(cat, on="episode_hash")
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=0)
    rows = df[["episode_hash", "zarr_processed_path", "right_act_start", "right_act_end"]].rename(
        columns={"right_act_start": "act_start", "right_act_end": "act_end"}).to_dict("records")
    print(f"fetching {len(rows)} trajectories")

    results = list(fetch_traj.map(rows, return_exceptions=True))
    good = [r for r in results if isinstance(r, dict) and r.get("ok")]
    print(f"{len(good)} ok, {len(results) - len(good)} failed")
    np.savez_compressed(out, hashes=np.array([r["episode_hash"] for r in good]),
                        **{r["episode_hash"]: r["traj"] for r in good})
    print(f"-> {out}")


# --------------------------------------------------------------------------- tier 3: pixels

# encoders to compare. All baked into the image so switching costs no cold-start download.
MODELS = {
    "dinov2-base":  "facebook/dinov2-base",       # 86M,  768d — the baseline
    "dinov2-large": "facebook/dinov2-large",      # 300M, 1024d
    "siglip2":      "google/siglip2-base-patch16-256",   # 375M, 768d, text-aligned
}
MODEL = MODELS["siglip2"]  # best within-episode progress signal; see PIPELINE.md 5b
FRAMES_PER_EPISODE = 32


def _bake_weights():
    from transformers import AutoImageProcessor, AutoModel

    for repo in MODELS.values():
        AutoImageProcessor.from_pretrained(repo)
        AutoModel.from_pretrained(repo)


gpu_image = (
    modal.Image.debian_slim()
    .uv_pip_install("boto3", "numpy", "torch", "torchvision", "transformers", "pillow",
                    "simplejpeg", "zstandard", "zarr==3.1.5", "scipy")
    .run_function(_bake_weights)
)


def project_hands(keypoints, head_pose, intrinsics):
    """World-frame MANO keypoints -> pixels. Mirrors egomimic's
    ActionChunkCoordinateFrameTransform (inverse of the head SE3) followed by
    pose_utils.cam_frame_to_cam_pixels. Quaternions are stored [qw,qx,qy,qz]."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    head_se3 = np.eye(4)
    head_se3[:3, :3] = Rotation.from_quat(head_pose[[4, 5, 6, 3]]).as_matrix()
    head_se3[:3, 3] = head_pose[:3]

    homogeneous = np.concatenate([keypoints, np.ones((len(keypoints), 1))], axis=1)
    cam = (np.linalg.inv(head_se3) @ homogeneous.T).T
    pixels = intrinsics @ cam.T
    return (pixels / pixels[2, :]).T[:, :2], cam[:, 2]


def hand_box(uv, depth, width: int, height: int, margin: float = 1.8):
    """Square crop around the visible hands, expanded so the manipulated object is included."""
    import numpy as np

    ok = (depth > 0.01) & (uv[:, 0] > -width) & (uv[:, 0] < 2 * width)
    if ok.sum() < 4:
        return None
    pts = uv[ok]
    centre = pts.mean(0)
    half = max(pts.max(0) - pts.min(0)) * margin / 2
    # a square box, shifted into bounds rather than clamped at both edges — clamping each
    # edge independently collapses the box to 1px when the hands project near the border
    side = float(np.clip(half * 2, 96, min(width, height)))
    x0 = float(np.clip(centre[0] - side / 2, 0, width - side))
    y0 = float(np.clip(centre[1] - side / 2, 0, height - side))
    return int(x0), int(y0), int(x0 + side), int(y0 + side)


def speed_profile(client, bucket: str, prefix: str, n: int):
    """Per-frame hand speed, from the pose arrays only (~0.25 MB, no pixels)."""
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import zarr

    work = Path(tempfile.mkdtemp())
    try:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix) + 1 :]
                if not rel or rel.startswith("images"):
                    continue
                target = work / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, obj["Key"], str(target))

        group = zarr.open_group(str(work), mode="r")
        keys = {name for name, _ in group.members(max_depth=None)}
        speeds, kps = [], []
        for side in ["left", "right"]:
            if f"{side}.obs_ee_pose" in keys:
                xyz = np.asarray(group[f"{side}.obs_ee_pose"][:n])[:, :3]
                speeds.append(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
            if f"{side}.obs_keypoints" in keys:
                kps.append(np.asarray(group[f"{side}.obs_keypoints"][:n]).reshape(n, 21, 3))

        intrinsics = dict(group.attrs).get("intrinsics", {}).get("front_1")
        return {
            "speed": np.maximum.reduce(speeds) if speeds else None,
            "keypoints": np.concatenate(kps, axis=1) if kps else None,
            "head": np.asarray(group["obs_head_pose"][:n]) if "obs_head_pose" in keys else None,
            "intrinsics": np.array(intrinsics) if intrinsics is not None else None,
        }
    except Exception:
        return {"speed": None, "keypoints": None, "head": None, "intrinsics": None}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def pick_frames(n: int, lo: int, hi: int, speed, count: int) -> list[int]:
    """Equal-motion spacing: each sampled frame covers the same amount of hand travel,
    so frames concentrate where something is actually happening instead of on dead air."""
    import numpy as np

    if speed is None or len(speed) < 4 or speed[lo:hi].sum() <= 0:
        return sorted(set(np.linspace(lo, max(hi, lo + 1), count).astype(int).tolist()))

    window = np.zeros_like(speed)
    window[lo:hi] = speed[lo:hi]
    cdf = np.cumsum(window)
    cdf = cdf / cdf[-1]
    targets = (np.arange(count) + 0.5) / count
    return sorted(set(np.searchsorted(cdf, targets).clip(0, n - 1).tolist()))


def fetch_frames(client, bucket: str, prefix: str, camera: str, wanted: list[int]):
    """Range-GET only the frames we need, using the shard index for byte offsets."""
    import numpy as np
    import simplejpeg
    import zstandard

    listing = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/{camera}/c/")
    shards = sorted(o["Key"] for o in listing.get("Contents", []))
    if not shards:
        raise FileNotFoundError(f"no shards under {prefix}/{camera}/c/")

    index = {}
    for shard_no, key in enumerate(shards):
        tail = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=-{INDEX_BYTES}")
        entries = np.frombuffer(tail["Body"].read()[:-4], dtype="<u8").reshape(INDEX_ENTRIES, 2)
        for local, (offset, nbytes) in enumerate(entries):
            if nbytes != EMPTY_CHUNK:
                index[shard_no * INDEX_ENTRIES + local] = (key, int(offset), int(nbytes))

    decompressor = zstandard.ZstdDecompressor()
    images, got = [], []
    for i in wanted:
        if i not in index:
            continue
        key, offset, nbytes = index[i]
        raw = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset + nbytes - 1}")
        payload = decompressor.decompress(raw["Body"].read())
        start = payload.find(b"\xff\xd8\xff")  # skip the vlen-bytes length header
        images.append(simplejpeg.decode_jpeg(payload[start:]))
        got.append(i)
    return images, got


@app.cls(
    image=gpu_image,
    gpu="L4",
    secrets=[modal.Secret.from_name("egoverse-r2")],
    max_containers=30,
    timeout=900,
)
class Embedder:
    encoder: str = modal.parameter(default="siglip2")

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        repo = MODELS[self.encoder]
        self.processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo)
        # SigLIP2 is a dual encoder; we only want the vision tower
        self.model = getattr(model, "vision_model", model).to("cuda").eval().half()
        self.torch = torch

    @modal.method()
    def embed(self, row: dict) -> dict:
        import os

        import numpy as np

        bucket = os.environ.get("BUCKET", "rldb")
        prefix = row["zarr_processed_path"].replace(f"s3://{bucket}/", "").rstrip("/")
        base = {"episode_hash": row["episode_hash"]}

        try:
            client = s3_client()
            n = int(row["n_frames"])
            lo = int(row.get("act_start", 0.0) * (n - 1))
            hi = int(row.get("act_end", 1.0) * (n - 1))
            bundle = speed_profile(client, bucket, prefix, n)
            speed = bundle["speed"] if row.get("motion_weighted", True) else None
            wanted = pick_frames(n, lo, max(hi, lo + 1), speed, FRAMES_PER_EPISODE)
            images, got = fetch_frames(client, bucket, prefix, "images.front_1", wanted)
            if not images:
                raise ValueError("no frames decoded")

            cropped = 0
            if row.get("hand_crop") and bundle["keypoints"] is not None \
                    and bundle["head"] is not None and bundle["intrinsics"] is not None:
                for k, t in enumerate(got):
                    h, w = images[k].shape[:2]
                    uv, depth = project_hands(bundle["keypoints"][t], bundle["head"][t],
                                              bundle["intrinsics"])
                    box = hand_box(uv, depth, w, h)
                    if box:
                        x0, y0, x1, y1 = box
                        images[k] = images[k][y0:y1, x0:x1]
                        cropped += 1

            batch = self.processor(images=images, return_tensors="pt").to("cuda")
            with self.torch.no_grad():
                out = self.model(pixel_values=batch["pixel_values"].half())
            emb = out.last_hidden_state[:, 0].float().cpu().numpy()  # CLS token
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            return {**base, "ok": True, "error": "", "frame_idx": got,
                    "cropped": cropped, "emb": emb.astype(np.float16)}
        except Exception as exc:
            return {**base, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


@app.local_entrypoint()
def embed(sample: int = 300, joined: str = "joined.parquet", out: str = "embeddings.npz",
          motion_weighted: bool = True, hand_crop: bool = False,
          encoder: str = "siglip2"):
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(joined)
    if "zarr_processed_path" not in df.columns:
        cat = pd.read_parquet("episodes.parquet")[["episode_hash", "zarr_processed_path"]]
        df = df.merge(cat, on="episode_hash")
    if "n_frames" not in df.columns:
        df["n_frames"] = df["num_frames"]
    for col, default in [("right_act_start", 0.0), ("right_act_end", 1.0)]:
        if col not in df.columns:
            df[col] = default
    df = df[df.n_frames.fillna(0) >= 16]
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=0)

    rows = df[["episode_hash", "zarr_processed_path", "n_frames",
               "right_act_start", "right_act_end"]].rename(
        columns={"right_act_start": "act_start", "right_act_end": "act_end"}).to_dict("records")
    for r in rows:
        r["motion_weighted"] = motion_weighted
        r["hand_crop"] = hand_crop
    print(f"embedding {len(rows)} episodes x {FRAMES_PER_EPISODE} frames on L4 "
          f"({'motion-weighted' if motion_weighted else 'uniform'}"
          f"{', hand-cropped' if hand_crop else ''}) with {MODELS[encoder]}")

    results = list(Embedder(encoder=encoder).embed.map(rows, return_exceptions=True))
    good = [r for r in results if isinstance(r, dict) and r.get("ok")]
    bad = [r for r in results if not (isinstance(r, dict) and r.get("ok"))]
    print(f"\n{len(good)} ok, {len(bad)} failed")
    for r in bad[:5]:
        print("  ", r if isinstance(r, Exception) else f"{r['episode_hash']}: {r['error']}")
    if hand_crop:
        tot = sum(r.get("cropped", 0) for r in good)
        print(f"hand-cropped {tot}/{sum(len(r['frame_idx']) for r in good)} frames")

    np.savez_compressed(
        out,
        hashes=np.array([r["episode_hash"] for r in good]),
        emb=np.stack([r["emb"].mean(0) for r in good]),
        frames=np.stack([r["emb"] for r in good if r["emb"].shape[0] == FRAMES_PER_EPISODE]),
        frame_hashes=np.array([r["episode_hash"] for r in good if r["emb"].shape[0] == FRAMES_PER_EPISODE]),
    )
    print(f"-> {out}")


@app.local_entrypoint()
def main(lab: str = "mecka", task: str = "cup_on_saucer", limit: int = 20,
         index: str = "episodes.parquet", out: str = "frames.parquet"):
    import pandas as pd

    catalog = pd.read_parquet(index)
    slice_ = catalog[
        ~catalog.is_deleted
        & catalog.lab.eq(lab)
        & catalog.task.eq(task)
        & catalog.zarr_processed_path.fillna("").str.strip().ne("")
    ]
    if limit:
        slice_ = slice_.sort_values("num_frames").iloc[:: max(1, len(slice_) // limit)].head(limit)
    rows = slice_[["episode_hash", "operator", "num_frames", "zarr_processed_path"]].to_dict("records")
    print(f"scanning {len(rows)} episodes from {lab}/{task}")

    results = list(scan_episode.map(rows, return_exceptions=True))
    good = [r for r in results if isinstance(r, dict) and r["ok"]]
    bad = [r for r in results if not (isinstance(r, dict) and r["ok"])]

    print(f"\n{len(good)} ok, {len(bad)} failed")
    for r in bad[:5]:
        print("  ", r if isinstance(r, Exception) else f"{r['episode_hash']}: {r['error']}")

    frames = pd.DataFrame(
        [
            {"episode_hash": r["episode_hash"], "frame_idx": i, "jpeg_bytes": n}
            for r in good
            for i, n in enumerate(r["sizes"])
        ]
    )
    frames.to_parquet(out, index=False)

    episodes = pd.DataFrame([{k: v for k, v in r.items() if k != "sizes"} for r in good])
    episodes.to_parquet(out.replace(".parquet", "_episodes.parquet"), index=False)

    mismatch = episodes[episodes.declared_frames.gt(0) & episodes.declared_frames.ne(episodes.n_frames)]
    print(f"\n{len(frames):,} frames -> {out}")
    print(f"{len(episodes)} episodes -> {out.replace('.parquet', '_episodes.parquet')}")
    print(f"frame-count mismatch vs SQL: {len(mismatch)}")
    print(episodes[["n_frames", "low_detail_frac", "faint_frac", "longest_faint_run", "size_cv", "min_z"]]
          .describe().loc[["min", "50%", "max"]].to_string())
