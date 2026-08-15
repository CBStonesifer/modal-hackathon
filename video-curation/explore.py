#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3", "sqlalchemy", "psycopg[binary]", "pandas", "pyarrow",
#   "zarr==3.1.5", "simplejpeg", "pillow", "numpy",
# ]
# ///
"""Access and explore EgoVerse. Pure Python — no AWS CLI, no repo checkout.

    ./explore.py setup                     # bootstrap creds -> ~/.egoverse_env
    ./explore.py index                     # dump app.episodes -> episodes.parquet + profile
    ./explore.py episode <episode_hash>    # pull one .zarr, print every key, save a frame
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import urllib.request
from pathlib import Path

ENV_FILE = Path.home() / ".egoverse_env"
README_URL = "https://raw.githubusercontent.com/GaTech-RL2/EgoVerse/main/README.md"
REGION = "us-east-2"
BUCKET = "rldb"
CATEGORICAL = ["lab", "task", "embodiment", "rig_name", "scene", "operator"]

# (admin secret, public read-only fallback) — mirrors egomimic/utils/aws/setup_secret.sh
R2_SECRETS = ("r2/rldb/credentials", "r2/rldb/public/credentials")
DB_SECRETS = ("rds/appdb/appuser", "rds/appdb/appuser-readonly")


# --------------------------------------------------------------------------- setup


def bootstrap_keys() -> tuple[str, str]:
    key = os.environ.get("EGOVERSE_AWS_KEY")
    secret = os.environ.get("EGOVERSE_AWS_SECRET")
    if key and secret:
        return key, secret

    # The bootstrap keys are published in the README on purpose; read them from there
    # so a rotation upstream fixes us for free and nothing lands in this repo.
    readme = urllib.request.urlopen(README_URL, timeout=30).read().decode()
    found = re.search(r"AccessKeyId:\s*(\S+)", readme), re.search(r"SecretAccessKey:\s*(\S+)", readme)
    if not all(found):
        raise SystemExit(
            f"could not find bootstrap keys in {README_URL}\n"
            "set EGOVERSE_AWS_KEY / EGOVERSE_AWS_SECRET from the README instead"
        )
    return found[0].group(1), found[1].group(1)


def resolve_secret(client, candidates: tuple[str, str]) -> tuple[str, str]:
    for name in candidates:
        try:
            response = client.get_secret_value(SecretId=name)
        except Exception:
            continue
        return name, response["SecretString"]
    raise SystemExit(f"could not read either of {candidates} — are the bootstrap keys still valid?")


def cmd_setup(args: argparse.Namespace) -> None:
    import boto3

    key, secret = bootstrap_keys()
    client = boto3.client(
        "secretsmanager",
        region_name=REGION,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )

    r2_name, r2_json = resolve_secret(client, R2_SECRETS)
    r2 = json.loads(r2_json)
    for required in ["access_key_id", "secret_access_key", "endpoint_url"]:
        if not r2.get(required):
            raise SystemExit(f"R2 secret {r2_name} is missing {required!r}")

    lines = {
        "R2_ACCESS_KEY_ID": r2["access_key_id"],
        "R2_SECRET_ACCESS_KEY": r2["secret_access_key"],
        "AWS_ENDPOINT_URL_S3": r2["endpoint_url"],
        "R2_ENDPOINT_URL": r2["endpoint_url"],
        "S3_ENDPOINT_URL": r2["endpoint_url"],
        "AWS_DEFAULT_REGION": REGION,
        "BUCKET": BUCKET,
    }
    db_name = None
    try:
        db_name, _ = resolve_secret(client, DB_SECRETS)
        arn = client.describe_secret(SecretId=db_name)["ARN"]
        lines["SECRETS_ARN"] = arn
        # aws_sql.py reads the DB secret at runtime, so the bootstrap keys must persist too
        lines["AWS_ACCESS_KEY_ID"] = key
        lines["AWS_SECRET_ACCESS_KEY"] = secret
    except SystemExit:
        print("! no DB secret resolved — `index` will not work, `episode --path` still will")

    ENV_FILE.write_text("".join(f"{k}={v}\n" for k, v in lines.items()))
    ENV_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)

    tier = "ADMIN" if r2_name == R2_SECRETS[0] else "PUBLIC read-only"
    print(f"\ntier      {tier}")
    print(f"R2 secret {r2_name}")
    print(f"DB secret {db_name or '(none)'}")
    print(f"endpoint  {r2['endpoint_url']}")
    print(f"\nwrote {ENV_FILE} (chmod 600)")


# --------------------------------------------------------------------------- clients


def load_env() -> None:
    if not ENV_FILE.exists():
        raise SystemExit(f"{ENV_FILE} missing — run ./explore.py setup first")
    for line in ENV_FILE.read_text().splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def db_engine():
    import boto3
    from sqlalchemy import URL, create_engine

    arn = os.environ.get("SECRETS_ARN")
    if not arn:
        raise SystemExit("SECRETS_ARN not set — the DB secret did not resolve during setup")
    payload = boto3.client("secretsmanager", region_name=REGION).get_secret_value(SecretId=arn)
    cfg = json.loads(payload["SecretString"])
    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=cfg.get("username", cfg.get("user")),
            password=cfg["password"],
            host=cfg["host"],
            port=cfg.get("port", 5432),
            database=cfg.get("dbname", "appdb"),
            query={"sslmode": "require"},
        ),
        pool_pre_ping=True,
    )


def r2_client():
    import boto3

    # R2 rejects the secret's session_token ("InvalidArgument: X-Amz-Security-Token") — it is
    # 40 chars, not an STS token. Key+secret alone work, and R2 wants region "auto".
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


# --------------------------------------------------------------------------- index


def cmd_index(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_sql("SELECT * FROM app.episodes", db_engine())
    out = Path(args.out)
    df.to_parquet(out, index=False)

    print(f"\n{len(df):,} rows x {len(df.columns)} columns -> {out}\n")
    print("COLUMNS")
    print(df.dtypes.to_string())

    print("\nNULL / EMPTY per column")
    for col in df.columns:
        blank = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        print(f"  {col:28s} {blank:>7,} missing")

    live = df[~df["is_deleted"]] if "is_deleted" in df else df
    print(f"\nlive episodes (is_deleted=False): {len(live):,}")

    for col in CATEGORICAL:
        if col not in live:
            continue
        counts = live[col].value_counts()
        print(f"\n{col.upper()} — {len(counts)} distinct")
        print(counts.head(args.top).to_string())

    if "num_frames" in live:
        print("\nNUM_FRAMES")
        print(live["num_frames"].describe().to_string())

    # is eval_success a real label, or just its default True? only the is_eval rows can tell us
    if {"is_eval", "eval_success"} <= set(live.columns):
        print("\nIS_EVAL x EVAL_SUCCESS")
        print(pd.crosstab(live["is_eval"], live["eval_success"]).to_string())
        print(f"\neval_score set (!= -1): {(live['eval_score'] != -1).sum():,}")


# --------------------------------------------------------------------------- episode


def lookup_episode(episode_hash: str) -> dict:
    import pandas as pd

    df = pd.read_sql(
        "SELECT * FROM app.episodes WHERE episode_hash = %(h)s",
        db_engine(),
        params={"h": episode_hash},
    )
    if df.empty:
        raise SystemExit(f"no row for {episode_hash}")
    return df.iloc[0].to_dict()


def download_prefix(s3_path: str, dest: Path, poses_only: bool = False, quiet: bool = False) -> Path:
    bucket = os.environ.get("BUCKET", BUCKET)
    prefix = s3_path.replace(f"s3://{bucket}/", "").lstrip("/").rstrip("/")
    client = r2_client()

    total = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            relative = Path(obj["Key"]).relative_to(prefix)
            if poses_only and relative.parts and relative.parts[0].startswith("images"):
                continue
            local = dest / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                client.download_file(bucket, obj["Key"], str(local))
            total += obj["Size"]
    if total == 0:
        raise SystemExit(f"nothing found at s3://{bucket}/{prefix}")
    if not quiet:
        print(f"pulled {total / 1e6:.1f} MB -> {dest}")
    return dest


def cmd_sample(args: argparse.Namespace) -> None:
    import pandas as pd

    df = pd.read_parquet(args.index)
    slice_ = df[
        ~df.is_deleted
        & df.lab.eq(args.lab)
        & df.task.eq(args.task)
        & df.zarr_processed_path.fillna("").str.strip().ne("")
    ]
    if slice_.empty:
        raise SystemExit(f"no live episodes for lab={args.lab} task={args.task}")

    # stratify by duration so the sample spans the range rather than clustering at the mode
    ranked = slice_.sort_values("num_frames").reset_index(drop=True)
    picks = ranked.iloc[:: max(1, len(ranked) // args.n)].head(args.n)

    cache = Path(args.cache)
    rows = []
    for i, row in enumerate(picks.itertuples(), 1):
        dest = cache / f"{row.episode_hash}.zarr"
        download_prefix(row.zarr_processed_path, dest, poses_only=args.poses_only, quiet=True)
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        rows.append({"episode_hash": row.episode_hash, "operator": row.operator,
                     "num_frames": row.num_frames, "mb": round(size / 1e6, 2)})
        print(f"[{i:>3}/{len(picks)}] {row.episode_hash}  {row.num_frames:>5.0f} frames  {size / 1e6:>6.2f} MB")

    manifest = cache / f"sample_{args.lab}_{args.task}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"\n{len(rows)} episodes, {sum(r['mb'] for r in rows):.0f} MB total -> {manifest}")


def describe_zarr(root: Path, frame_out: Path) -> None:
    import numpy as np
    import zarr

    group = zarr.open_group(str(root), mode="r")

    print("\n=== zarr.json attrs ===")
    print(json.dumps(dict(group.attrs), indent=2, default=str)[:4000])

    print("\n=== arrays ===")
    image_arrays = []
    for name, node in sorted(group.members(max_depth=None)):
        if not hasattr(node, "shape"):
            continue
        print(f"  {name:34s} {str(node.shape):16s} {str(node.dtype):14s} chunks={node.chunks}")
        if name.startswith("images"):
            image_arrays.append((name, node))
        elif node.ndim == 2 and node.shape[0] > 1:
            row = np.asarray(node[0])
            head = np.array2string(row[:7], precision=4, suppress_small=True)
            print(f"      [0] = {head}{' …' if row.size > 7 else ''}")

    if "obs_rgb_timestamps_ns" in group:
        stamps = np.asarray(group["obs_rgb_timestamps_ns"][:])
        valid = stamps[stamps > 0]
        declared_frames = group.attrs.get("total_frames")
        declared_fps = group.attrs.get("fps")
        true_fps = len(valid) / ((valid[-1] - valid[0]) / 1e9)

        print(f"\nallocated {len(stamps)} rows, {len(valid)} real (total_frames={declared_frames})")
        if len(valid) != len(stamps):
            print(f"  ! rows {len(valid)}: are zero padding — slice every array to total_frames")
        print(f"duration {(valid[-1] - valid[0]) / 1e9:.1f} s -> true fps {true_fps:.1f}, attrs say {declared_fps}")
        if declared_fps and abs(true_fps - declared_fps) / declared_fps > 0.1:
            print("  ! declared fps is wrong — derive it from obs_rgb_timestamps_ns")

    if "annotations" in group:
        print("\n=== annotations ===")
        spans = group["annotations"][...]
        for raw in spans:
            print(" ", raw if isinstance(raw, str) else bytes(raw).decode())
        if len(spans) == 0:
            print("  (none — this episode has no language spans)")

    if not image_arrays:
        return
    import simplejpeg
    from PIL import Image

    name, arr = image_arrays[0]
    # scalar indexing on a zarr v3 object array returns an 8-byte pointer view, not the blob
    blob = arr[arr.shape[0] // 2 : arr.shape[0] // 2 + 1][0]
    img = simplejpeg.decode_jpeg(bytes(blob))
    Image.fromarray(img).save(frame_out)
    print(f"\nmid frame of {name}: {len(blob):,} B JPEG -> {img.shape} {img.dtype} -> {frame_out}")


def cmd_episode(args: argparse.Namespace) -> None:
    row = lookup_episode(args.episode_hash)
    print("\n=== SQL row ===")
    for key, value in row.items():
        print(f"  {key:26s} {value}")

    cache = Path(args.cache)
    if args.mp4 and row.get("zarr_mp4_path"):
        download_prefix(row["zarr_mp4_path"], cache / f"{args.episode_hash}.mp4")
        return

    dest = download_prefix(row["zarr_processed_path"], cache / f"{args.episode_hash}.zarr", args.poses_only)
    describe_zarr(dest, cache / f"{args.episode_hash}.png")


# --------------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    setup = sub.add_parser("setup", help="exchange bootstrap keys for R2 + DB credentials")
    setup.set_defaults(func=cmd_setup, needs_env=False)

    index = sub.add_parser("index", help="dump and profile app.episodes")
    index.add_argument("--out", default="episodes.parquet")
    index.add_argument("--top", type=int, default=25)
    index.set_defaults(func=cmd_index, needs_env=True)

    episode = sub.add_parser("episode", help="pull and describe one episode")
    episode.add_argument("episode_hash")
    episode.add_argument("--cache", default="./data")
    episode.add_argument("--mp4", action="store_true", help="fetch the preview MP4 instead")
    episode.add_argument("--poses-only", action="store_true", help="skip the image arrays")
    episode.set_defaults(func=cmd_episode, needs_env=True)

    sample = sub.add_parser("sample", help="pull a duration-stratified sample of a slice")
    sample.add_argument("--lab", required=True)
    sample.add_argument("--task", required=True)
    sample.add_argument("-n", type=int, default=24)
    sample.add_argument("--poses-only", action="store_true")
    sample.add_argument("--index", default="episodes.parquet")
    sample.add_argument("--cache", default="./data")
    sample.set_defaults(func=cmd_sample, needs_env=True)

    args = parser.parse_args()
    if args.needs_env:
        load_env()
    args.func(args)


if __name__ == "__main__":
    main()
