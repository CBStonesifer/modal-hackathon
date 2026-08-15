# Handoff

EgoVerse curation engine (hackathon Track 1). Everything runs. Read this, then `RESULTS.md`.

**One-line status:** the pipeline is complete end-to-end and produces `keep_drop.csv` for 5,407
episodes. The one thing missing is validation against human labels, because no ground truth
exists in the dataset.

---

## 1. Get running in five minutes

```bash
./explore.py setup          # credentials -> ~/.egoverse_env. Pure Python, no AWS CLI needed.
./explore.py index          # -> episodes.parquet (446,957 rows). ~1 min.
```

Modal is already configured on this machine: profile `cbstonesifer`, secret `egoverse-r2`.
On a new machine:

```bash
uv tool install modal --with pandas --with pyarrow    # pandas is for @app.local_entrypoint
modal token set --token-id ... --token-secret ... --profile=<name> --activate
set -a; . ~/.egoverse_env; set +a
modal secret create egoverse-r2 \
  R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  R2_ENDPOINT_URL="$R2_ENDPOINT_URL" BUCKET=rldb
```

Full pipeline:

```bash
modal run curate.py::main   --limit 0                # tier 1, 66 s
modal run curate.py::poses  --limit 0                # tier 2, 6m23s
modal run curate.py::embed  --sample 0 --hand-crop   # tier 3, GPU (SigLIP2 default), ~40 min
modal run curate.py::trajectories --sample 0         # -> traj.npz, for the ablation
./score.py --keep-frac 0.70                          # -> keep_drop.csv
./ablate.py --horizon 15                             # -> ablation.csv
```

`modal shell curate.py::scan_episode` drops you inside the container to debug imports and paths.

---

## 2. What each file is

| file | purpose |
|---|---|
| **`RESULTS.md`** | the submission write-up. Start here after this doc |
| **`DATA.md`** | what the dataset actually contains — measured, not documented |
| **`PIPELINE.md`** | every experiment including the failures. The lab notebook |
| `explore.py` | credentials, catalog dump, single-episode inspection, sampling |
| `curate.py` | the Modal app — tiers 1, 2, 3 |
| `score.py` | feature selection, scoring, failure taxonomy, quotas, trim spans → `keep_drop.csv` |
| `ablate.py` | proxy-policy ablation. Its result is negative — see §5 item 4 |
| `EGOVERSE_NOTES.md`, `MODAL_NOTES.md` | background research, pre-dates the build |
| `SYSTEM_DESIGN.md`, `CONCEPTS.md` | early design and explainer. **Superseded by `DATA.md`** — their illustrative numbers were guesses (720×1280, 342 MB) and the real values are 360–480×640 and 12–27 MB |

### Data artifacts

| file | rows | what |
|---|---|---|
| `episodes.parquet` | 446,957 × 21 | the whole SQL catalog, every lab and task |
| `frames.parquet` | 904,511 × 3 | per-frame JPEG byte size — tier 1 output |
| `frames_episodes.parquet` | 5,407 × 13 | per-episode visual features |
| `kinematics.parquet` | 5,407 × 44 | per-episode motion features — tier 2 |
| `joined.parquet` | 5,407 × 56 | the two merged + operator. **Score inputs come from here** |
| `emb_cup_crop.npz` | 400 | hand-cropped DINOv2, 32 frames each |
| `emb_multiscene.npz` | 400 | the 58-scene control experiment |
| `multiscene.parquet` | 14,742 | the multi-scene slice definition |
| `keep_drop.csv` | 5,407 | **the deliverable** |

⚠ `embeddings.npz` was overwritten by a 6-episode smoke test. The original 16-frame baseline
numbers survive only in `PIPELINE.md`. Regenerate with
`modal run curate.py::embed --sample 400 --no-hand-crop` if you need it live.

---

## 3. Things that are settled — do not re-derive

1. **Capture identity dominates every naive signal.** JPEG size 84% demonstrator, hand span 84%,
   duration 70%, jerk RMS 43%. Measured via η². This is the central result.
2. **Use SPARC, not jerk.** Same property, 24% vs 43% confounded.
3. **Cross-episode visual comparison fails on single-scene slices and cannot be repaired.** Four
   attempts — more frames, motion weighting, frame-delta, hand-cropping — none got below 85%
   nearest-neighbour-shares-demonstrator against 8.9% chance.
4. **It works when scenes vary.** On 58 scenes the task effect is 2.4× the scene effect. The
   confound is the data's lack of variety, not the model.
5. **Within-episode analysis is immune** — demonstrator is constant, so it cancels exactly.
   Progress curves, trim spans, retry detection are all safe.
6. **The progress U-curve is real**, reproduced across two sampling schemes and two samples.
   Hand-cropping deepens the dip 32%.
7. **Visual and kinematic quality are uncorrelated** (|r| ≤ 0.09). Both are needed.
8. **The keypoint→pixel transform** is `inv(head_SE3) @ p_world`, quaternion stored `[qw,qx,qy,qz]`
   so scipy needs `[4,5,6,3]`. Implemented in `curate.py::project_hands`, verified visually in
   `data/keypoint_projection.png`.
9. **No threshold is intrinsic to the data.** The proxy ablation has a 1.3–2.6% total dynamic
   range and quality ranking captures ~1% of it at 1.2σ; the score distribution is unimodal with
   no knee. Any keep fraction is a budget decision — say so rather than calling it optimal.
10. **A bigger encoder does not fix the confound** (98.0% → 95.2% across a 4.4× parameter range),
    but **SigLIP2 is the best of the three** on within-episode progress and is now the default.
    The axis for a real improvement is modality (video-native), not size.
11. **Five failure axes are independent** (max pairwise correlation 0.14), and k-means over the
    drop population recovers exactly one cluster per axis. Every drop is named; 15.4% of ranked
    drops are honestly labelled `below_average` — no identifiable defect.

---

## 4. Traps that cost time

| trap | fix |
|---|---|
| R2 secret carries a 40-char `session_token` | ignore it — R2 rejects it. Key+secret alone work |
| R2 wants `region_name="auto"` | `us-east-2` is for Secrets Manager only |
| Newest `zarr` won't open these stores | pin `zarr==3.1.5` |
| `arr[200]` on an image array returns an 8-byte pointer | use `arr[200:201][0]` |
| Pose arrays over-allocated (500 rows, 413 real) | slice to `total_frames`; padding is zeros, but *images* past the cutoff are still decodable |
| Declared `fps` can be wrong by 2× (Aria said 30, timestamps said 15.4) | derive from `obs_rgb_timestamps_ns`. **mecka has none**, so its 30 is unverifiable |
| `zarr.json` metadata is unreliable — mecka says `task_name: "debug"` | read all metadata from Postgres |
| `sync_s3.py`'s `DATA_FILTERS` presets match **zero rows** | they filter `embodiment == 'aria'`; vendor moved to the `lab` column on 2026-07-08 |
| `modal run curate.py` is ambiguous once there are several entrypoints | use `curate.py::main` |
| `@app.local_entrypoint` runs locally and needs pandas | `uv tool install modal --with pandas --with pyarrow` |

---

## 5. What is NOT done

In priority order. Each entry says what it *buys*, not just what is missing — they exist for
different reasons and only two of them change what you can claim.

| # | item | depends on | effort | what it changes |
|---|---|---|---|---|
| 1 | human labels | — | ~30 min human | method → measured result |
| 2 | full tier-3 pass | — | ~30 min compute | removes imputation, fixes one drop count |
| ~~3~~ | ~~proxy-policy ablation~~ | — | **done** | built; result is negative — see below |
| ~~4~~ | ~~justify the keep rate~~ | — | **answered** | no threshold is derivable — see below |
| 5 | score the multi-scene slice | — | ~1 hr | generality + adds diversity |
| 6 | hand-crop on multi-scene | — | ~7 min | possible simplification |

**Item 1 is now the only outstanding thing that changes what you can claim.** Items 3 and 4 were
completed and both returned negative results, which are recorded rather than hidden.

### 1. No validation against human labels

*Purpose: convert the engine from a **method** into a **result**.*

Everything it does is justified by construction — features selected by measured confounding,
scores normalised within demonstrator, quotas applied per person. That makes the drops
*defensible*, not *correct*. Nobody has confirmed that an episode the engine calls bad is bad.

This is also the only check that could catch a systematic error in the whole chain. Every other
item assumes the engine is roughly right; this is the one that tests it. And it is the answer to
a judge's first question — "how do you know it works?" — which currently has no number.

There is no ground truth to borrow: `is_eval` is `False` on all 420,415 rows and `eval_success`
is its dataclass default. Fix: pull ~150 preview MP4s (`zarr_mp4_path`, ~2 MB each), label
usable/unusable, report precision/recall **on held-out demonstrators** — a random split leaks,
because near-duplicate retakes from the same person land on both sides.

### 2. Tier 3 covers 400 of 5,407 episodes

*Purpose: stop imputing two of the fourteen scoring features.*

`progress_dip` and `progress_final` are median-filled for 93% of the slice. Scores survive it —
twelve real features outweigh two fabricated ones — but the `no_visible_change` hard-drop rule
can only fire where progress data exists. **The current count of 22 is drawn from a 7.4% sample,
implying roughly 290 across the full slice**, so one of the five drop reasons is under-counted by
an order of magnitude.

Purely mechanical. A full pass was launched and had not finished at handoff — check
`/tmp/emb_all.log` and `emb_all.npz`; `./score.py` picks it up automatically once present.

### 3. Proxy-policy ablation — DONE, negative result

*Purpose was: test whether curation actually helps, which nothing else here does.*

**Built as `ablate.py`. It cannot discriminate.** The first proxy (next-frame prediction) was
saturated — identical MSE at 313 and 3,131 episodes, because a 15-parameter linear model learns
"hands move smoothly" from 300 episodes. Longer horizons plus 1024 random Fourier features gave
data quantity a **1.3–2.6%** total effect, of which quality ranking captured ~1% at 1.2σ. Not
significant at any horizon or keep fraction.

To make this discriminate you need the *visual* BC task — embedding → action — where the model
must infer intent from the scene rather than extrapolate smooth physics. That needs poses paired
with the sampled frames, which tier 3 does not currently return. Original rationale below, still
valid as motivation:

Every other measurement describes properties of the **data**. This one measures whether the
selected subset trains a better policy than a random subset of equal size. Different claims, and
only the second is what curation is for.

It is also the rebuttal to the sharpest available objection: *"maybe the dropped episodes were
fine, and throwing away 30% of the data just makes things worse."* Nothing in the repo can
currently answer that.

The brief names this deliverable explicitly ("a validation report using proxy metrics"). Intended
chart: held-out action-prediction error of a small behaviour-cloning MLP trained on random-N vs
quality-N vs quality+diversity-N. All inputs exist — pose arrays are the actions, embeddings are
cached — it is simply unbuilt.

### 4. The keep rate — ANSWERED: it cannot be derived

*Purpose was: turn a budget knob into a recommendation.*

**There is no threshold intrinsic to this data.** The score distribution is unimodal (one
histogram maximum, knee at 92%), the ablation is flat, and calibration against the 63
independently-verifiable bad episodes is weak — their median percentile is 37.3% against 50.1%
for everything else, and cutting the bottom 10% catches only 27% of them.

Defensible cuts, in descending order of confidence: the **63 hard drops**; the **257 episodes
(4.8%) failing two independent axes**, which is 28% above the 201 expected by chance; and beyond
that any percentile, explicitly labelled a budget decision. Original rationale below:

63 episodes are dropped for measurable defects; 1,612 are dropped because someone picked 70%.
That second number is a decision wearing the costume of a finding.

Item 3 fixes it: sweep `--keep-frac`, plot held-out policy error against fraction kept, find
where the curve flattens. That chart says "keep 62%, and here is why." **Cannot be done without
item 3.**

### 5. The multi-scene slice has not been scored

*Purpose: prove generality, and recover the half of the deliverable that is currently missing.*

`multiscene.parquet` — 14,742 episodes, 75 scenes, 83 tasks — has only had the embedding
experiment run on it. Two payoffs. The obvious one is generalisation: the engine currently works
on one task in one kitchen.

The larger one: **cross-episode diversity selection should work there** (task effect 2.4× scene
effect, measured). The current engine contains no diversity term at all, because diversity
provably failed on `cup_on_saucer`. Scoring multiscene would let you add it back with evidence —
and curation is quality *and* coverage, so half the story is missing without it.

Largest upside to the scope of the result.

### 6. Hand-cropping untested on multi-scene data

*Purpose: a clean symmetric experiment, and a possible simplification.*

Cropping fails to remove the demonstrator confound in a single room because the hands *are* the
person. But where scenes vary it removes the **room** — a different confound — and might sharpen
the task signal directly. If it works it could replace frame-delta, which currently buys a 91:1
task/scene ratio at the cost of a 9× weaker signal; cropping might get both.

Lowest priority: it optimises something that already works, and it is a clean yes/no in about
seven minutes.

---

## 6. If you only do one thing

Label 150 episodes from the preview MP4s and report precision/recall on held-out demonstrators.

Everything in the current engine is *principled* — feature selection is driven by measured
confounding rather than taste — but nothing in it is *validated*. That one step converts
"defensible method" into "measured result", and it costs about half an hour of watching 2 MB
clips.

---

## 7. Framing, if you present this

The strongest claim is not the manifest. It is that **the obvious signals are mostly measuring
capture identity, and none of them fails loudly.** Frame size, duration, jerk, embedding
similarity — each produces a plausible number and sensible-looking clusters while ranking
demonstrators instead of demonstrations.

EgoVerse's own paper argues that scale alone is insufficient and that alignment between human
data and the robot's objective is what matters. Showing that the natural proxies for that
alignment are confounded — with η² for each, and four failed repair attempts — is directly
responsive to the gap the authors name.

Lead with the confound table. The keep/drop list is the consequence, not the point.
