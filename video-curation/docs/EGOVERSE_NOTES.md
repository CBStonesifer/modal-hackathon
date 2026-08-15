# EgoVerse — Dataset Reference

Source: https://github.com/GaTech-RL2/EgoVerse · https://egoverse.ai · arXiv:2604.07607

---

## 1. What it is

EgoVerse is a **collaborative platform for egocentric human demonstration data for robot learning** — dataset + codebase + multi-lab empirical study. Not a static dump: the repo ships the data schema, S3 sync tooling, format converters, a training stack (PyTorch Lightning + Hydra), baseline policies (ACT, EgoMimic/HPT, Pi), and a web episode inspector.

Published by **Georgia Tech's Robot Learning and Reasoning Lab (RL2)**, PI Danfei Xu, as a consortium with Stanford (REAL), UC San Diego (Wang Lab), ETH Zürich, Meta Reality Labs, Mecka AI, Scale AI, MicroAGI, Lightwheel, Trace Labs.

Two tiers: **EgoVerse-A** (standardized protocols mirrored across academic labs) and **EgoVerse-I** (industry-sourced, larger, more in-the-wild).

**Scale** (paper, April 2026): 1,362 hours · ~80,000 episodes · 1,965 tasks · 240 scenes · 2,087 demonstrators. The project site now claims 4,003 hours — it's continuously expanding, so treat the paper's numbers as a floor.

**Repo vitals:** MIT license (code), CC BY-SA 4.0 (data), 519 stars, 147 open issues, created Feb 2025, actively pushed as of Aug 2026. 844 tracked files.

### The paper's central claim — and why it's the hackathon's opening

> Policy performance generally improves with more human data, but effective scaling requires **alignment between the human data and the robot's learning objective**. Scale alone is not sufficient.

The authors' own finding is that naive scale underperforms. Every one of the three hackathon tracks is a direct response to a gap the paper itself identifies and the repo does **not** currently fill.

---

## 2. Data format

Per-episode **Zarr v3** store, directory named `<episode_hash>.zarr/` where the hash is a UTC timestamp `YYYY-MM-DD-HH-MM-SS-ffffff`.

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `images.front_1` | `(T,)` varlen bytes | JPEG | egocentric RGB, **per-frame JPEG blobs, not a video container** |
| `images.left_wrist` / `right_wrist` | `(T,)` varlen bytes | JPEG | optional |
| `obs_head_pose` | `(T,7)` | float64 | `[tx,ty,tz,qw,qx,qy,qz]`, SLAM world frame, meters |
| `left.obs_ee_pose` / `right.obs_ee_pose` | `(T,7)` | float64 | end-effector pose |
| `left.obs_wrist_pose` / `right.obs_wrist_pose` | `(T,7)` | float64 | |
| `left.obs_keypoints` / `right.obs_keypoints` | `(T,63)` | float64 | 21 MANO hand landmarks × 3, flattened |
| `left.obs_gripper` / `right.obs_gripper` | `(T,1)` | float64 | aperture in [0,1], optional |
| `annotations` | `(N,)` | JSON | span-based language labels, optional |
| `obs_rgb_timestamps_ns` | `(T,)` | int64 | UTC nanoseconds |

`zarr.json` attrs: `embodiment`, `total_frames`, `fps` (30 or 60), `task_name`, `task_description`, **mandatory** `intrinsics` (`{camera_key: 3×4 K}`), optional `extrinsics`, `features`.

**Annotation schema:**
```json
{"text": "pick up the shirt", "start_idx": 0, "end_idx": 145}
```
Sub-step imperative English. Spans may overlap and need not cover the whole episode.

**Embodiments:** `human_right_arm`, `human_left_arm`, `human_bimanual` (ids 1–3); robot `eva_*` (4–6). Vendor tags (`aria_*`, `mecka_*`, `scale_*`, `lightwheel_*`) were removed from the embodiment field on 2026-07-08 and now live only in a SQL `lab` column — a breaking change that invalidates pre-change local caches.

**Not present:** no audio, no depth, **no success/failure label**, no fixed train/val manifest, no documented total size on disk, no standardized resolution (varies by capture device).

Training-time convention: everything is re-expressed relative to `obs_head_pose` into a head-centric frame (+X right, +Y down, +Z forward).

---

## 3. Getting the data — the credential flow

**Two-stage.** AWS credentials → AWS Secrets Manager → R2 credentials → the bucket. The keys in the README are *bootstrap* credentials for Secrets Manager only; they are not the storage credentials, and publishing them is deliberate.

### Setup (once, locally)

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && ./aws/install -i ~/aws-cli -b ~/bin

aws configure          # keys are in the README; region us-east-2

./egomimic/utils/aws/setup_secret.sh
```

### What `setup_secret.sh` does

Calls `aws secretsmanager get-secret-value` against `us-east-2`, trying an admin secret then falling back to a public read-only one:

| Purpose | Admin secret | Public fallback |
|---|---|---|
| R2 storage | `r2/rldb/credentials` | `r2/rldb/public/credentials` |
| Postgres index | `rds/appdb/appuser` | `rds/appdb/appuser-readonly` |

Prints `Downloading EgoVerse Admin Credentials` or `Downloading Public EgoVerse read only credentials` depending on which resolved. The public tier is what the README keys are scoped to — that fallback is the intended path for external users.

The R2 secret is JSON with `access_key_id`, `secret_access_key`, optional `session_token`, and `endpoint_url`. **The endpoint comes from the secret — never hardcode it.**

Writes `~/.egoverse_env` (chmod 600):

```
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_SESSION_TOKEN=...            # only if present
AWS_ENDPOINT_URL_S3=https://<account>.r2.cloudflarestorage.com
R2_ENDPOINT_URL=<same>
S3_ENDPOINT_URL=<same>
AWS_DEFAULT_REGION=us-east-2
BUCKET=rldb
SECRETS_ARN=...                 # only if the DB secret resolved
```

Overridable via env vars: `REGION`, `DB_SECRET_NAME`, `PUBLIC_DB_SECRET_NAME`, `R2_SECRET_NAME`, `PUBLIC_R2_SECRET_NAME`, `ENV_FILE`, `BUCKET`.

### How runtime code picks it up

`egomimic/utils/aws/aws_data_utils.py`:

```python
def load_env(path="~/.egoverse_env", required: bool = False):
    ...
    os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))
```

Reads `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_SESSION_TOKEN`, each falling back to the `AWS_*` equivalent. Missing file = warning, or raises if `required=True`.

### ★ Implication for Modal

`setdefault` means **already-set environment variables win over the file.** So don't recreate `~/.egoverse_env` in a container — run `setup_secret.sh` once locally, then push those values into a `modal.Secret`:

```bash
modal secret create egoverse-r2 \
  R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
  AWS_ENDPOINT_URL_S3=... R2_ENDPOINT_URL=... BUCKET=rldb
```
```python
@app.function(secrets=[modal.Secret.from_name("egoverse-r2")])
```

No AWS CLI, no AWS keys, and no Secrets Manager round-trip inside your containers. `load_env()` becomes a harmless no-op. **Caveat:** if the R2 secret includes a `session_token` those credentials are temporary — check once, and if so plan to re-run `setup_secret.sh` and refresh the Modal Secret.

### Storage and index

- **Storage:** Cloudflare R2 (S3-compatible), bucket `rldb`, paths `s3://rldb/processed_v3/<prefix>/<episode_hash>.zarr/`. Older data under `raw_v2`/`processed_v2`.
- **Index:** hosted PostgreSQL, table `app.episodes` (hash, operator hash, lab, task, embodiment, zarr path, frame count). Filters query it. See `sql_tutorial.ipynb`.
- **Download:**
  ```bash
  python egomimic/scripts/data_download/sync_s3.py --local-dir <dir> --filters aria-fold-clothes
  ```
  CLI args: `--local-dir` (required), `--filters`, `--workers` (default 128). Presets in `DATA_FILTERS`: `aria-fold-clothes`, `aria-all`, `eva-all`, `mecka-fold-clothes`. Shells out to **`s5cmd`** (Go binary — `apt_install` or direct download on Modal, not pip).
- The training pipeline downloads data automatically; `sync_s3.py` is the manual path.
- **Browse without downloading:** https://partners.mecka.ai/egoverse — viewer only.

**Unofficial HF mirrors** (third-party, unverified provenance/fidelity, LeRobot format):
`Jay-Ye/EgoVerse-Aria-LeRobot-V3.0` (~82.7 GB), `Jay-Ye/EgoVerse-Scale-LeRobot-V3.0`, `Jay-Ye/EgoVerse-Mecka-LeRobot-V3.0`, plus assorted `dreamzero-egoverse-*` uploads. There is **no official GaTech-RL2 org on HuggingFace** (404).

**Documented target composition:** ~85% tabletop manipulation / ~10% mobile manipulation / ~5% navigation; ~60% parallel-jaw / ~40% either gripper; ~70% staged studio / ~30% in-the-wild.

---

## 4. Repo structure — what's worth reading

```
egomimic/
├── trainHydra.py                     # ENTRYPOINT: Hydra + Lightning training
├── rldb/                             # === DATASET CORE ===
│   ├── zarr/zarr_dataset_multi.py    # 72KB — ZarrDataset, MultiDataset, S3EpisodeResolver
│   ├── zarr/zarr_writer.py           # 31KB — write the canonical schema
│   ├── zarr/hdf5_to_zarr.py          # ALOHA HDF5 -> Zarr
│   ├── embodiment/{embodiment,human,eva}.py   # per-embodiment transforms, keymaps, viz
│   └── filters.py                    # DatasetFilter -> S3EpisodeResolver
├── scripts/
│   ├── embedding_process/            # ★ DINOv3 / Qwen3 embedding generation for zarr
│   ├── aria_process/                 # Aria .vrs -> zarr (aria_utils.py, 41KB)
│   ├── mecka_process/, eva_process/  # vendor -> zarr
│   ├── language_process/             # Scale AI annotation pipeline + LLM prompts
│   ├── data_download/sync_s3.py      # the download path
│   ├── data_visualization/           # inspector_lib/ — web episode browser (views.py 80KB)
│   ├── ray_helper.py                 # Ray cluster parallel processing
│   └── tutorials/                    # dataset_tutorial.ipynb, embodiment_tutorial.ipynb, zarr_data_viz.ipynb
├── algo/                             # act.py, hpt.py (EgoMimic), pi.py
├── hydra_configs/                    # data/*.yaml, model/*.yaml, launcher/submitit*.yaml (SLURM)
├── robot/                            # ROS2 + ARX arms + RealSense + Quest teleop — IGNORE for our purposes
└── utils/aws/                        # RDS+S3 SQL utils, budget guardrails

external/
├── lerobot/                          # full vendored HuggingFace LeRobot copy
├── openpi/                           # git submodule -> GaTech-RL2/openpi (pi0/pi0.5)
└── scale/scripts/                    # Scale sensor-fusion -> zarr
```

**★ Start here:** `egomimic/scripts/embedding_process/` already generates **DINOv3 and Qwen3 embeddings** for episodes. Tracks 1 and 2 are both embedding-geometry problems — this is half the pipeline already written, in the dataset's native format. Read it before writing your own embedder.

**Also read first:** `egomimic/scripts/tutorials/dataset_tutorial.ipynb` and `zarr_data_viz.ipynb` — fastest way to see a real episode's contents.

### Setup

```bash
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```
Python ≥3.11. Key deps: `torch==2.6.0`, `zarr==3.1.5`, `transformers==4.57.3`, `simplejpeg`, `boto3`, `sqlalchemy`, `psycopg`, `s5cmd`, `decord`, `av==12.0.0`, `projectaria-tools[all]==2.0.0`, `lightning`, `hydra-core`, `ray`.

`requirements.txt` and `pyproject.toml` **disagree** on pins (pyproject has torch 2.7.1, transformers 4.53.2). Pick one — requirements.txt is what the README uses.

### What NOT to drag onto Modal

- **The repo's `Dockerfile`** — it's `FROM ros:humble-ros-base` and builds a ROS2 workspace + C++ robot-arm SDK + CAN bus + USB device passthrough. That's robot hardware control, irrelevant to dataset work. Build a clean image from `requirements.txt` instead.
- **`mujoco-py`** (legacy, Cython build against libGL/libosmesa/patchelf, breaks headless). Modern `mujoco==3.4.0` is fine but you likely need neither.
- **`projectaria-tools[all]`** — heavy, native/GUI deps. Only needed if converting raw `.vrs`; skip it if you're reading processed Zarr.
- **`rospkg`, `pyrealsense2`, `libgl1`/`libegl1`/`libusb`** — live-capture stack.
- **SLURM/submitit launchers** (`submitit_pace.yaml`, `submitit_skynet.yaml`) — tied to GT's PACE/Skynet clusters. Modal replaces this entirely.
- **`external/openpi`** — private-ish submodule, needs `--recursive`, plus GCS `gsutil` access for pi0.5 checkpoints and a JAX→PyTorch conversion.

`s5cmd` needs `apt_install` or a binary download in your Modal image — it's not a pip package.

---

## 5. Where EgoVerse sits

| Dataset | Contents | Annotations |
|---|---|---|
| **Ego4D** | 3,600+ hrs, 926 wearers, 74 locations | Dense narrations (`#C C does...`), verb/noun taxonomy, hand-object state-change frames, 110-class moments |
| **Ego-Exo4D** | 1,422 hrs, synced ego+exo, 131 scenes | Keystep taxonomy, proficiency levels, object masks, expert commentary, Aria MPS head pose/gaze |
| **EPIC-KITCHENS-100** | Kitchen egocentric | Verb–noun segments (97 verbs × 300 nouns), hand-object boxes |
| **EgoDex** (Apple) | 829 hrs, Vision Pro, 194 tasks | 3D upper-body pose, 25 keypoints/hand, 30Hz 1080p. CC-BY-NC-ND |
| **Open X-Embodiment** | 22 robots, 2M+ trajectories, 527 skills | RLDS/TFRecord, varying action spaces |
| **DROID** | 76k teleop trajectories, 564 scenes | 3 stereo streams, NL instructions. **Pre-filtered to successes** |
| **HOT3D** (Meta) | 833 min, Aria + Quest 3 | Mocap GT 3D hand poses (UmeTrack + MANO), object poses, gaze |
| **Aria Everyday Activities** | 143 recordings, 7.3 hrs | Time-aligned speech-to-text, eye gaze, MPS point clouds |

EgoVerse's differentiator: **multi-institution, multi-vendor, standardized-schema, continuously growing, and paired with robot data on a matched embodiment (Eva)** for direct transfer study.

Note that DROID and most demo datasets are **pre-filtered to successes** — there is very little natural failure data in the field. That's context for how novel track 3's prevalence audit is.

---

## 6. Prior art per track

### Track 1 — The Curation Engine

Named methods worth citing (and beating, or borrowing from):

- **DemInf** (arXiv 2502.08623) — rank trajectories by estimated mutual information I(S;A) via VAE embeddings. Demonstrates 50%-subset-at-parity.
- **CUPID** (cupid-curation.github.io) — influence functions estimating each demo's *causal* effect on closed-loop policy performance.
- **SCIZOR** (arXiv 2505.22626) — self-supervised, targets two failure modes: suboptimal frames (progress-prediction filter) and redundant segments (dedup).
- **SIEVE** (arXiv 2607.06442) — structure-aware selection for VLAs; reports beating SCIZOR.
- **Smoothness metrics** (arXiv 2604.23000) — third-order jerk, **Spectral Arc Length (SAL)**, **Trajectory-Envelope Distance (TED)**. ★ EgoVerse gives you `obs_ee_pose` and `obs_keypoints` at 30–60fps — these are computable **without touching a GPU**.
- **CLIP-score filtering** — cosine sim between clip embedding and `task_description`. LAION thresholds ~0.3 / top 15–30%.
- **Submodular / facility-location** — the classical coverage method; `apricot` implements it. **SubZeroCore** (arXiv 2509.21748) is training-free and closed-form — good under time pressure.
- **`score_lerobot_episodes`** (HuggingFace) — existing episode-QC scorer (visual clarity, motion artifacts, idle time). EgoVerse converts to LeRobot, so this is nearly plug-and-play as a baseline.

**Cheapest strong angle:** kinematic quality (jerk/SAL/TED on pose arrays, CPU-only, seconds per episode) + embedding-coverage selection. You get a defensible keep/drop signal without a GPU pass, then use embeddings only for the diversity half.

### Track 2 — Quantitative Diversity

The brief explicitly says "aside from text" — so this is embedding geometry. Ranked by cost:

| Method | What it computes | Cost |
|---|---|---|
| **k-means cell coverage + Shannon/Simpson** | Entropy over cluster occupancy | Cheapest, most demoable (bar chart + one number). Sensitive to k |
| **Mean pairwise cosine distance** | Avg (1 − cos sim) over pairs | O(n²) or subsampled. Crude — ignores clustering structure |
| **Vendi Score** (arXiv 2210.02410, `pip install vendi-score`) | exp(Shannon entropy of eigenvalues of similarity kernel K) | **O(d²n)** with low-dim embeddings — very tractable. No reference set needed |
| **DPP log-determinant** | log-det of kernel submatrix = volume spanned | Also gives you greedy MAP subset *selection* with (1−1/e) guarantee |
| **Density & Coverage** (Naeem et al. ICML 2020, `clovaai/generative-evaluation-prdc`) | Coverage = fraction of reference samples with a candidate in their k-NN sphere | Needs a reference distribution + k-NN index (FAISS). O(n·m) |

**Recommended:** Vendi Score as the headline number (principled, one scalar, cites cleanly, cheap), k-means occupancy as the visual, DPP log-det if you also want to *select* the diverse subset rather than just score it. Vendi is the strongest "score that ranks two subsets" answer.

### Track 3 — The Human Reward Model

- **SuccessVQA** (arXiv 2303.07280, "VLMs as Success Detectors") — prompt a VLM with final frame(s) + "was task X completed?". Benchmarked on Ego4D. Simple, but **final-frame-only misses mid-episode failures** and states that look identical (a closed drawer that was the wrong drawer).
- **VIP** (arXiv 2210.00030, `facebookresearch/vip`) — ★ self-supervised value/progress embedding **trained on Ego4D human video**, no action labels. Closest domain match to EgoVerse. ResNet-50 scale, <2GB VRAM. Distance-to-goal-embedding = dense progress signal.
- **LIV** (arXiv 2306.00958, `penn-pal-lab/LIV`) — VIP + language conditioning, trained on EPIC-KITCHENS. Ships `load_liv()`.
- **RoboCLIP** (arXiv 2310.07899) — reward = alignment between rollout video embedding and one reference demo/text embedding. Zero-shot from a single demo, cross-embodiment.
- **Progress prediction** — monotonic regressor over frames; episodes whose progress never nears 1.0 or regresses are flagged. (This is SCIZOR's internal mechanism.)
- **Trajectory-endpoint heuristics** (arXiv 2605.20388) — hand/camera trajectory shape often separates success from failure with no visual model at all. Cheap, but needs per-task calibration.

**Known failure modes:** (1) final-state ambiguity; (2) demo data is success-biased, so there's little failure data to learn from; (3) human-hand↔robot-gripper domain gap degrades VIP/LIV/RoboCLIP cross-embodiment; (4) VLM classifiers are prompt-sensitive and weak on partial failures — multi-frame beats single-final-frame (arXiv 2508.18705).

**Recommended:** VIP progress curve over the episode gives you a *confidence meter over a video segment* essentially for free, which is literally one of the named deliverables. Pair with a small logistic head on SigLIP2 embeddings trained on a few hundred hand-labels for the binary flag.

---

## 7. Models to run

**Image/frame embedding**
- `google/siglip2-so400m-patch14-384` — ~400M params, ~2–3GB VRAM fp16. Solid default. Use `-224` for cheaper.
- Repo already uses **DINOv3** in `embedding_process/` — check what it produces before adding a second embedder.

**Video-native embedding**
- `OpenGVLab/InternVideo2-CLIP-1B-224p-f8` — 8-frame clips, real temporal signal. ~4–6GB fp16 (estimate, verify).

**Progress / reward**
- VIP and LIV checkpoints — not on HF Hub; auto-download via `load_vip()` / `load_liv()`. <2GB.

**Video-language QA**
- `Qwen/Qwen2.5-VL-7B-Instruct` — ~16GB fp16 (A10/L40S), ~8–9GB in 4-bit.
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct` — ~5.2GB. Smaller: `SmolVLM2-500M-Video-Instruct` (1.8GB), `-256M-` (1.38GB).

**Success classifier**
- No off-the-shelf robot-success checkpoint exists on HF. Expect to train a small head: SigLIP2/InternVideo2 embeddings → logistic regression or 2-layer MLP on a few hundred hand-labeled clips. Trains in seconds, near-zero marginal VRAM, and is far easier to explain to a judge than VLM prompting.

---

## 8. Practical notes for building on this

1. **Frames are JPEG blobs in Zarr, not video files.** No ffmpeg decode needed — `simplejpeg.decode_jpeg` on the bytes. Faster and simpler than you'd expect. Random frame access is free.
2. **Pose arrays are tiny and CPU-only.** `(T,7)` float64 per stream. Kinematic quality metrics over 80k episodes are cheap enough to run without a GPU at all — do this pass first, it may be your whole track-1 signal.
3. **`intrinsics` is mandatory in `zarr.json`** — you can do real 3D geometry, not just 2D.
4. **Cache decoded embeddings on a Modal Volume, keyed by episode hash.** Compute once, then score arbitrarily many candidate subsets for free. This is the single highest-leverage architectural decision for tracks 1 and 2.
5. **Don't return feature tensors from Modal functions** — >2MiB round-trips through us-east. Write to the Volume, return scalars.
6. **Run `setup_secret.sh` first, before writing any code.** It's self-serve — the README keys bootstrap the public read-only tier. Confirm which tier you land on from its output, then move those env vars into a Modal Secret (§3).
7. **Scope to one filter preset.** `aria-fold-clothes` is a bounded, coherent slice — a single task with many demonstrators is *ideal* for both diversity comparison and success/failure labeling, because task is held constant. Don't try to process all 80k episodes.
