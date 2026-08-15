"""Preview footage for the curation manifest: R2 -> Modal Volume -> web endpoint.

    modal run clips.py::sync --limit 20      # smoke test
    modal run --detach clips.py::sync        # all 5,407 clips, ~11 GB
    modal run clips.py::build_manifest       # keep_drop.csv + catalog -> manifest.json
    modal deploy clips.py                    # serve both

The clips already exist in R2 as ~2 MB `_video.mp4` siblings of each Zarr store. Copying them
into a Volume is what lets a browser reach them: the UI gets one public origin instead of R2
credentials, and the endpoint is the same for the manifest and the footage.
"""

from __future__ import annotations

import re

import modal

app = modal.App("egoverse-clips")

volume = modal.Volume.from_name("egoverse-footage", create_if_missing=True)
VOLUME_PATH = "/footage"
CLIPS_DIR = f"{VOLUME_PATH}/clips"
MANIFEST_PATH = f"{VOLUME_PATH}/manifest.json"

HASH_PATTERN = re.compile(r"^[0-9a-f]{24}$")

fetch_image = modal.Image.debian_slim().uv_pip_install("boto3")
web_image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]")


@app.function(
    image=fetch_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("egoverse-r2")],
    max_containers=50,
    retries=modal.Retries(max_retries=3),
    timeout=600,
)
def fetch_clip(row: dict) -> dict:
    import os
    from pathlib import Path

    import boto3

    episode_hash = row["episode_hash"]
    destination = Path(CLIPS_DIR) / f"{episode_hash}.mp4"
    if destination.exists():
        return {"episode_hash": episode_hash, "ok": True, "bytes": destination.stat().st_size}

    bucket = os.environ.get("BUCKET", "rldb")
    key = row["zarr_mp4_path"].replace(f"s3://{bucket}/", "")

    try:
        client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".part")
        client.download_file(bucket, key, str(partial))
        partial.rename(destination)
    except Exception as exc:
        return {"episode_hash": episode_hash, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {"episode_hash": episode_hash, "ok": True, "bytes": destination.stat().st_size}


def load_manifest_rows(manifest: str, catalog: str) -> list[dict]:
    import pandas as pd

    decisions = pd.read_csv(manifest)
    fields = ["episode_hash", "zarr_mp4_path", "task", "scene", "num_frames"]
    catalog_rows = pd.read_parquet(catalog, columns=fields).drop_duplicates("episode_hash")

    merged = decisions.merge(catalog_rows, on="episode_hash", how="left")
    missing = merged.zarr_mp4_path.isna().sum()
    if missing:
        print(f"warning: {missing} episodes have no preview mp4 in the catalog")
    return merged.to_dict("records")


@app.local_entrypoint()
def sync(manifest: str = "keep_drop.csv", catalog: str = "episodes.parquet",
         limit: int = 0, drops_only: bool = False):
    rows = load_manifest_rows(manifest, catalog)
    rows = [r for r in rows if isinstance(r.get("zarr_mp4_path"), str)]
    if drops_only:
        rows = [r for r in rows if r["decision"] == "drop"]
    if limit:
        rows = rows[:limit]

    print(f"syncing {len(rows)} clips into volume 'egoverse-footage'")
    results = list(fetch_clip.map(rows))

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"{len(ok)} ok, {len(failed)} failed, {sum(r['bytes'] for r in ok) / 1e9:.2f} GB")
    for r in failed[:10]:
        print(f"  {r['episode_hash']}: {r['error']}")


@app.local_entrypoint()
def build_manifest(manifest: str = "keep_drop.csv", catalog: str = "episodes.parquet"):
    import json
    import math
    import tempfile

    stored = stored_hashes()
    episodes = []
    for row in load_manifest_rows(manifest, catalog):
        record = {k: (None if isinstance(v, float) and math.isnan(v) else v)
                  for k, v in row.items() if k != "zarr_mp4_path"}
        record["has_clip"] = record["episode_hash"] in stored
        episodes.append(record)

    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(episodes, handle)
        handle.flush()
        with volume.batch_upload(force=True) as batch:
            batch.put_file(handle.name, "/manifest.json")

    with_clips = sum(1 for e in episodes if e["has_clip"])
    print(f"manifest.json: {len(episodes)} episodes, {with_clips} with clips")


def stored_hashes() -> set[str]:
    try:
        entries = volume.listdir("clips")
    except FileNotFoundError:
        return set()
    return {entry.path.rsplit("/", 1)[-1].removesuffix(".mp4") for entry in entries}


@app.function(image=web_image, volumes={VOLUME_PATH: volume}, max_containers=4)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    web_app = FastAPI()
    # read-only public data; the UI is a static site with no fixed origin
    web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                           allow_headers=["*"])

    @web_app.get("/manifest.json")
    def manifest():
        import os

        if not os.path.exists(MANIFEST_PATH):
            raise HTTPException(404, "manifest not built — run clips.py::build_manifest")
        return FileResponse(MANIFEST_PATH, media_type="application/json")

    @web_app.get("/clip/{episode_hash}.mp4")
    def clip(episode_hash: str):
        import os

        if not HASH_PATTERN.match(episode_hash):
            raise HTTPException(400, "bad episode hash")
        path = f"{CLIPS_DIR}/{episode_hash}.mp4"
        if not os.path.exists(path):
            raise HTTPException(404, "clip not in volume")
        # FileResponse honours Range, so the browser can seek
        return FileResponse(path, media_type="video/mp4")

    return web_app
