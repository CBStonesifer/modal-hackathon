# What you're actually working with

Everything below is **measured**, not read off documentation. Where the docs and the data
disagree, the data wins and the disagreement is flagged. Reproduce with `./explore.py`.

```bash
./explore.py setup                                  # bootstrap creds -> ~/.egoverse_env
./explore.py index                                  # -> episodes.parquet + full profile
./explore.py episode 2025-10-13-03-55-42-588000     # pull one .zarr, dump keys, save a frame
```

Pure Python. **No AWS CLI and no repo checkout** — `explore.py` does the Secrets Manager
exchange itself with boto3, so nothing is written to your `~/.aws`. It is a `uv run --script`
file: dependencies install on first run.

---

## 1. Access — verified working

Two hops: published bootstrap keys → AWS Secrets Manager → real R2 + Postgres credentials.

```
tier      PUBLIC read-only
R2 secret r2/rldb/public/credentials
DB secret rds/appdb/appuser-readonly
endpoint  https://1beb594fb475d71c4420f7b693524e19.r2.cloudflarestorage.com
```

The README's AWS keys are bootstrap credentials for Secrets Manager only and are published
deliberately. `explore.py setup` scrapes them from the README at run time, so a rotation
upstream fixes you for free and no credential lands in this repo. Override with
`EGOVERSE_AWS_KEY` / `EGOVERSE_AWS_SECRET` if that ever breaks.

**Three things that will waste your afternoon:**

| Trap | Fix |
|---|---|
| The R2 secret carries a 40-char `session_token`. Passing it → `InvalidArgument: X-Amz-Security-Token` | Ignore it. Key + secret alone work |
| R2 wants `region_name="auto"`, not `us-east-2` | `us-east-2` is for Secrets Manager only |
| Latest `zarr` rejects these stores — `chunk edge length must be >= 1, got 0` | Pin `zarr==3.1.5`, as the repo does |

---

## 2. The index — `app.episodes`

**446,957 rows, 439,053 live, 21 columns.** Note that is ~5.5× the paper's ~80,000 episodes;
the table has grown well past the publication.

The `TableRow` dataclass in `egomimic/utils/aws/aws_sql.py` documents 17 columns. The live
table has **four more it doesn't mention**: `license`, `segments`, `created_at`, `updated_at`.

| Column | What it is | Populated | Notes |
|---|---|---|---|
| `episode_hash` | UTC timestamp `YYYY-MM-DD-HH-MM-SS-ffffff` | 100% | PK, and a real capture time |
| `lab` | data source, 9 distinct | ~100% | the confounder in everything |
| `task` | free text, **27,997 distinct** | ~100% | not a taxonomy — see below |
| `task_description` | free-text instruction | 98% | the only prose field |
| `embodiment` | `human_bimanual` etc., 6 distinct | ~100% | determines which pose keys exist |
| `operator` | demonstrator, 4,588 distinct | **20%** | missing on all of microagi |
| `rig_name` | capture rig, 6 distinct | **20%** | `aria_gen1`, `mecka`, `eva`, `abc_yam`, `scale` |
| `scene` | environment, 402 distinct | **15%** | free diversity axis |
| `objects` | Postgres array, e.g. `{"gray vneck t-shirt"}` | **11%** | free diversity axis |
| `segments` | **language spans in SQL** | 5% | ★ see below |
| `num_frames` | length | ~100% | **unreliable — see §3** |
| `zarr_processed_path` | `s3://rldb/processed_v3/<lab>/<hash>.zarr` | ~100% | |
| `zarr_mp4_path` | preview MP4 | 96% | ★ watch episodes without Zarr |
| `zarr_processing_error` | non-empty = conversion failed | 54 rows | drop these |
| `license` | e.g. `CC BY-SA 4.0` | 14% | |
| `is_deleted` | soft delete | 100% | **always filter** (7,904 rows) |
| `is_eval` / `eval_score` / `eval_success` | — | — | ★ **dead, see below** |
| `created_at` / `updated_at` | timestamps | 100% | undocumented; useful for staleness |

### ★ There are no success labels. At all.

I checked, because it would have changed which track is cheapest:

```
IS_EVAL x EVAL_SUCCESS          eval_score
is_eval  False -> 420,415 rows  all -1 (unset)
         True  ->       0 rows  eval_success all True (the dataclass default)
```

`is_eval` is `False` on every row, `eval_success` is its default `True`, `eval_score` is its
default `-1`. These three columns are schema placeholders that were never populated.

**Consequence:** track 3 has no ground truth to bootstrap from and no prevalence number to look
up. Every label is one you create. That makes it the highest-effort track — and also the one
where a few hundred hand-labels constitute genuinely novel data, which is the whole premise of
the track. Budget for labelling, and use the preview MP4s to do it.

### ★ `segments` — language annotations without downloading anything

23,289 rows carry a Postgres array of sub-step spans, at **second** resolution:

```json
[{"label": "pick up screwdriver",          "start_seconds": 0.0,  "end_seconds": 6.4},
 {"label": "place screwdriver on keyboard","start_seconds": 6.4,  "end_seconds": 8.6},
 {"label": "connect cable to laptop",      "start_seconds": 8.6,  "end_seconds": 21.4}]
```

This is the same content as the Zarr `annotations` array but reachable from SQL. You can mine
sub-step structure across 23k episodes for the price of one query. 56% of `mecka` rows have it;
essentially no other lab does.

### ★ The dataset is 80% one lab, and that lab has almost no metadata

| lab | episodes | scene | objects | operator | mp4 | segments |
|---|---|---|---|---|---|---|
| **microagi** | 355,926 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| mecka | 41,617 | 0.86 | 1.00 | 1.00 | 1.00 | **0.56** |
| abc | 18,456 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| scale | 17,090 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 |
| rl2 | 4,811 | 1.00 | 1.00 | 1.00 | 0.96 | 0.00 |
| eth | 543 | 1.00 | 1.00 | 1.00 | 0.99 | 0.00 |
| song / yam / wang | 610 | mixed | | | | |

*(fraction of rows where the column is non-empty)*

`microagi` is 81% of the corpus and a metadata desert — no operator, no scene, no objects, no
rig name. `mecka` is the richest large lab. `rl2` (Georgia Tech, the flagship Aria capture) is
small but complete.

This is the single most important shape fact in the dataset. Any "diversity" number computed
over everything is mostly a statement about microagi. Any metadata-driven filter silently drops
to the 19% tail. **Pick your slice deliberately and say so.**

### `task` is free text, not a taxonomy

27,997 distinct strings over 439k episodes. `fold_clothes` (12,617) and `fold_laundry` (9,044)
are separate values. So are `ironing_clothes` (3,800) and `iron_clothes` (2,353). Top tasks:

```
wash_dishes 41,800 · fold_clothes 12,617 · fold_laundry 9,044 · cup_on_saucer 6,673
prepare_vegetables 5,110 · pack_green_peppers 4,752 · ironing_clothes 3,800
```

Near-duplicate task names are themselves a curation finding — and a cheap first win for track 1.

### The repo's download filters are broken

`sync_s3.py`'s `DATA_FILTERS` all filter `row['embodiment'] == 'aria'`. Vendor tags were removed
from `embodiment` on 2026-07-08 and moved to `lab`. **Every preset there now matches zero rows**,
including `aria-fold-clothes`. Write your own:

```python
live = df[~df.is_deleted & (df.zarr_processing_error == "")]
slice_ = live[(live.lab == "rl2") & (live.task == "fold_clothes")]     # 936 episodes
```

---

## 3. Inside an episode

Verified on `2025-10-13-03-55-42-588000` — `lab=rl2`, `rig_name=aria_gen1`,
`embodiment=human_bimanual`, `task=fold_clothes`, `objects={"short sleeve shirts","long sleeve
shirts",shorts}`. **27.2 MB** on disk.

```
annotations                (0,)        object    chunks=(1,)
images.front_1             (500,)      object    chunks=(1,)     JPEG blobs, ~53 KB each
left.obs_aria_keypoints    (500, 63)   float64   chunks=(100,63)
left.obs_ee_pose           (500, 7)    float64
left.obs_keypoints         (500, 63)   float64
left.obs_wrist_pose        (500, 7)    float64
obs_eye_gaze               (500, 3)    float64                   ← eye gaze, undocumented
obs_head_pose              (500, 7)    float64
obs_rgb_timestamps_ns      (500,)      int64
right.*                    (same as left)
```

Pose rows are `[tx, ty, tz, qw, qx, qy, qz]`; keypoints are 21 MANO landmarks × 3.
Frames decode to **(480, 640, 3) uint8** — not 720p. Resolution varies by rig; check, don't assume.

Differences from the documented schema: **`obs_eye_gaze` exists** (Aria gaze vector, nobody
mentions it), there are **two keypoint variants** (`obs_keypoints` and `obs_aria_keypoints`),
there is **no gripper array** for human embodiments, and `annotations` is often `(0,)` — empty.

### ★ Arrays are over-allocated. Slice to `total_frames`.

```
allocated 500 rows, 413 real (total_frames=413)
```

Rows past `total_frames` are zero padding in every pose array. The **image** array still holds
decodable JPEGs past the cutoff, so a naive "sample 16 frames uniformly over the array" pulls
frames that no pose or timestamp corresponds to. Slice everything to `total_frames` first.

### ★ The declared `fps` is wrong

```
zarr.json says fps = 30
obs_rgb_timestamps_ns says 413 frames over 26.9 s = 15.4 fps
```

Off by 2×. Any velocity, jerk, or smoothness metric that divides by the declared fps is wrong by
a factor of two — and wrong *inconsistently across labs*, which means you build a rig detector
instead of a quality metric. **Always derive fps from `obs_rgb_timestamps_ns`.** `explore.py`
prints both and flags the mismatch.

`num_frames` in SQL (413) happens to match `total_frames` here, but neither matches the array
length. Trust `total_frames` for validity and timestamps for timing.

### ★ Scalar indexing on the image array is broken

```python
arr[200]              # -> 8-byte ndarray (a pointer view). Garbage.
arr[200:201][0]       # -> bytes, 54,543 B, magic ffd8ffe0. Correct.
```

A zarr v3 varlen-object array returns a pointer view under scalar indexing. Always slice.

### ★ Per-frame quality without downloading frames

The image array is **sharded**, not one chunk per frame:

```json
"chunk_grid": {"chunk_shape": [1000]},
"codecs": [{"name": "sharding_indexed", "configuration": {
    "chunk_shape": [1], "codecs": ["vlen-bytes", "zstd"], "index_location": "end"}}]
```

All frames live in one object (`images.front_1/c/0`, 11.7 MB) with a `1000 × (offset, nbytes)`
u64 index in the **last 16,004 bytes**. A ranged GET of that tail yields every frame's
compressed size without transferring any pixels:

```python
body = s3.get_object(Bucket="rldb", Key=shard_key, Range="bytes=-16004")["Body"].read()
entries = np.frombuffer(body[:-4], dtype="<u8").reshape(1000, 2)
sizes = entries[entries[:, 1] != 2**64 - 1][:, 1]     # exact, 20 B constant overhead
```

Verified: `corr(index_length, actual_jpeg_bytes) = 1.00000`, 156 ms per episode.

And JPEG size is a usable sharpness proxy — against per-frame Laplacian variance:

| rig | corr(bytes, laplacian) | corr(bytes, luminance) | blurriest frame |
|---|---|---|---|
| mecka | **+0.953** | +0.177 | smallest of 170 |
| aria_gen1 | +0.520 | +0.502 | smallest of 413 |

In both, the blurriest frame was the *smallest*. The proxy is strong within mecka's fixed
camera and scene; on Aria it is diluted by exposure variation, so **normalise within rig and
within episode**, and treat it as a relative signal, not an absolute sharpness score.

**Cost for all 5,409 `cup_on_saucer` episodes (~920k frames): 87 MB of range reads, versus
63 GB to download the shards.** This makes a per-frame pass over an entire slice essentially free.

### The split that makes this affordable

Pose arrays are ~1 MB; images are the other ~26 MB (and far more at higher resolutions). They
are separate subdirectories in the store, so a pose-only pass over thousands of episodes costs
almost nothing and needs no GPU. Jerk, path length, spectral arc length, idle fraction, duration
— all computable without decoding a single frame.

### Two more silent-failure traps

- **Poses are in SLAM world coordinates.** Absolute position encodes *which room*, not what the
  hands did. Convert to the head-centric frame (repo convention: +X right, +Y down, +Z forward)
  before comparing episodes.
- **Quaternions are sign-ambiguous.** `q` and `−q` are the same rotation, so naive differencing
  produces phantom spikes. Use `2 · arccos(|⟨q_t, q_{t+1}⟩|)`.

---

## 4. Rigs differ more than tasks do

A second episode, `692e9f8d39719ab57395b838` — `lab=mecka`, `task=cup_on_saucer`, 170 frames,
**12.0 MB**. It differs from the Aria one in ways that break shared code:

| | aria_gen1 (rl2, eth, song, wang) | mecka |
|---|---|---|
| `episode_hash` | UTC timestamp `2025-10-13-03-55-42-588000` | ObjectId `692e9f8d39719ab57395b838` |
| resolution | 480 × 640 | 360 × 640 |
| `obs_rgb_timestamps_ns` | present | **absent** |
| zero padding past `total_frames` | yes (500 alloc / 413 real) | none — exact |
| `obs_eye_gaze`, `obs_aria_keypoints` | present | absent |
| `zarr.json` `task_name` | correct (`fold_clothes`) | **`"debug"`**, description empty |

Two consequences:

- **`zarr.json` is not trustworthy metadata. SQL is.** The mecka store calls its task `"debug"`.
  Read `task`, `lab`, `operator` from `app.episodes`, never from the store.
- **You cannot verify fps on mecka.** No timestamps, so you must take the declared 30 on faith —
  and the declared fps was measurably wrong (30 vs 15.4) on the Aria episode. Any cross-rig
  smoothness comparison inherits that risk. Compare within a rig, or state the assumption.

---

## 5. What kinds of episodes exist — choosing a slice

439k live episodes, but they fall into five populations that have almost nothing in common.
Numbers are human embodiments only unless stated.

| Population | n | Shape | Metadata | Use for |
|---|---|---|---|---|
| **mecka — atomic clips** | 41,617 | 144–233 frames (~6 s), one repetition | 20+ operators, objects 100%, `segments` on 56% | ★ controlled curation |
| **aria_gen1 — long sessions** | 2,543 | 2,000–10,000 frames (1–6 min), multi-step | complete: operator, scene, objects | rich but small |
| **microagi — wild bulk** | 355,926 | 165–1,200 frames, max 34,001 | **none** — no operator/scene/objects/rig | scale, no controls |
| **scale — freeform/flagship** | 17,090 | 180–6,000 frames | 1 operator, 1 scene (bulk tags) | tasks prefixed `freeform_*` / `flagship_*` |
| **robot (eva, yam)** | 21,776 | — | — | not human demos |

### The same task name means different things per lab

`fold_clothes` exists in 6 labs (12,212 episodes) and is the obvious cross-lab target — until
you look at length:

| lab | n | operators | p10 | median | p90 | max |
|---|---|---|---|---|---|---|
| mecka | 1,607 | 20 | 144 | **195** | 233 | 726 |
| microagi | 9,896 | 0 | 169 | **325** | 1,189 | 34,001 |
| rl2 | 572 | 20 | 1,946 | **2,777** | 3,744 | 8,208 |
| eth | 76 | 1 | 2,333 | **2,933** | 10,029 | 10,870 |

A mecka `fold_clothes` is one fold. An rl2 `fold_clothes` is a 3-minute laundry session — 14×
longer. They are not the same task and should not share a quality threshold. (`mecka` also has
`folding_clothes` as a *separate* task value with 350 episodes. The 27,997 free-text task names
are full of this.)

### Where the curation opportunity actually is

Fraction of each `fold_clothes` slice that is anomalous by length alone:

```
microagi  n=9896   <150 frames: 0.1%    >1500 frames: 6.8%   max 34,001
mecka     n=1607   <150 frames: 11.9%   >1500 frames: 0.0%   max 726
rl2       n= 572   <150 frames: 0.0%    >1500 frames: 98.6%  max 8,208
```

microagi has 673 runaway episodes; mecka has 191 suspiciously truncated ones. Both are keep/drop
candidates you can find with **zero downloads** — the length signal is already in the parquet.

### `segments` is mecka-only and lives on other tasks

All 23,289 episodes with SQL language spans are `mecka`, and none are on `fold_clothes` or
`cup_on_saucer`. They sit on `dishwashing` (882), `cleaning_shoes` (742), `sharpening_knives`
(562), `potting_plants` (458), `folding_clothes` (350), `ironing_clothes` (316).

### Download sizing

Measured: ~70 KB per frame with images, ~1 MB per episode for poses alone.

| Slice | episodes | frames | with pixels | poses only |
|---|---|---|---|---|
| mecka `cup_on_saucer` | 5,409 | 0.9 M | ~65 GB | ~5 GB |
| mecka `fold_clothes` | 1,607 | 316 k | ~20 GB | ~1.6 GB |
| rl2 `fold_clothes` | 572 | 1.7 M | ~108 GB | ~0.6 GB |
| microagi `fold_clothes` | 9,896 | 6.4 M | ~415 GB | ~10 GB |
| all `fold_clothes` | 12,212 | 9.0 M | ~582 GB | ~12 GB |

Pixels for anything but a sample are out of reach in a hackathon. Poses for any of these are
trivial. That asymmetry should drive the architecture: kinematics over the whole slice,
embeddings over a subset.

---

## 6. What is not there

No audio. No depth. **No success/failure labels** (§2). No fixed train/val manifest. No
standardised resolution. `annotations` is empty on most episodes — the language that does exist
is mostly in SQL `segments`, concentrated in `mecka`.

Everything else in the repo — `egomimic/algo/`, `robot/`, the ROS 2 Dockerfile,
`external/openpi` — is training and hardware infrastructure, irrelevant to data work.
