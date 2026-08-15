# Track 1 build plan — fetch, process, curate, output

Target slice: `lab=mecka`, `task=cup_on_saucer`, **5,409 live episodes**, 22 operators,
~920k frames. All costs below are measured, not estimated.

---

## 1. Fetching — four tiers, each one filters what the next must touch

Modal is rented machines, not a model provider. Tiers 0–2 need **no GPU and no model**.

| Tier | What you fetch | Bytes | Per episode | Whole slice on Modal |
|---|---|---|---|---|
| **0 · metadata** | `app.episodes` via Postgres | 40 MB | — | done, local |
| **1 · shard index** | last 16,004 B of each image shard | **87 MB** | 156 ms | ~20 s, 50 CPU containers |
| **2 · poses** | every array except `images*` | **1.5 GB** | ~1 s | ~2 min, 50 CPU containers |
| **3 · frames** | 32 ranged reads per surviving episode | ~2.2 MB/ep | ~3 s | ~20 min, GPU |

Tier 3 is 5.2× cheaper than it looks because the tier-1 index gives byte offsets for **every
individual frame**, so you range-GET only the 32 you sample:

```python
idx  = s3.get_object(Bucket=B, Key=shard, Range="bytes=-16004")["Body"].read()
ent  = np.frombuffer(idx[:-4], dtype="<u8").reshape(1000, 2)     # (offset, nbytes)

off, n = ent[i]                                                   # any frame, random access
raw    = s3.get_object(Bucket=B, Key=shard, Range=f"bytes={off}-{off+n-1}")["Body"].read()
jpeg   = zstandard.ZstdDecompressor().decompress(raw)[8:]         # vlen-bytes header
img    = simplejpeg.decode_jpeg(jpeg)                             # (360, 640, 3) uint8
```

### Modal shape

```python
r2 = modal.Secret.from_name("egoverse-r2")      # R2_ACCESS_KEY_ID / SECRET / ENDPOINT_URL
vol = modal.Volume.from_name("egoverse-feat", create_if_missing=True)

@app.function(secrets=[r2], volumes={"/feat": vol}, max_containers=50,
              retries=modal.Retries(max_retries=3))
def scan_index(episode_hash: str, shard_key: str) -> dict:
    ...
    np.save(f"/feat/{episode_hash}_frames.npy", sizes)    # write to the Volume
    return {"hash": episode_hash, "n": len(sizes), "blur_frac": float(...)}   # scalars only
```

Three rules from `MODAL_NOTES.md` that matter here: never return arrays >2 MiB (they route
through us-east — write to the Volume and return scalars); every write path must start with the
mount prefix or it silently lands on ephemeral disk; use `.map(..., return_exceptions=True)` so
one bad episode doesn't kill the run. Cap `max_containers` so you don't hammer R2.

---

## 2. The one model you actually need

Nothing in tiers 0–2 uses a model. Tier 3 uses exactly one.

**VIP** — `facebookresearch/vip`, ResNet-50, ~1 GB, `load_vip()` (auto-downloads, not on HF Hub).
Self-supervised on **Ego4D human egocentric video**, which is the closest domain match that
exists to EgoVerse. Give it frames, get a 1024-d embedding each. Progress at frame *t* is the
negative distance from `emb[t]` to the goal embedding `emb[T]`. No labels, no fine-tuning, no
prompt engineering. This is the only thing that can answer *"does this episode demonstrate the
full action"* without you labelling first.

**SigLIP2** — `google/siglip2-so400m-patch14-384`, 1152-d — optional, only if you want
near-duplicate retake detection or a diversity term in selection. Skip it for the minimum build.

Everything else — the scorer, the selector, the trimmer — is your numpy. That's the project.

---

## 3. Curating — features, then a decision

### Per-episode feature vector

| Source | Features | Cost |
|---|---|---|
| tier 0 | `num_frames`, `operator` | free |
| tier 1 | `blur_frac`, `size_cv`, `min_z`, `longest_dark_run`, `max_step_drop` | 87 MB |
| tier 2 | `jerk_rms`, `sal`, `path_len`, `idle_frac`, `bimanual_asym` | 1.5 GB |
| tier 3 | `progress_final`, `progress_max`, `monotonicity`, `plateau_len`, `regression_depth` | GPU |

Tier-1 features come from the per-frame byte series. Because the camera and scene are fixed
within a mecka episode, variation over time is mostly motion blur and focus — so z-score the
series **within the episode**, and call frames below −2σ blurred.

### The decision, in three passes

```python
# 1. hard drops — data integrity, not taste
drop_if(num_frames < 60)                       # 2 episodes: cannot contain the task
drop_if(blur_frac > 0.25)                      # a quarter of frames unusable
drop_if(longest_frozen_run > 15)               # capture stalled

# 2. per-operator normalisation — THE confounder fix
#    operator 6871d82a's median clip is 108 frames, 690362600's is 236.
#    A global threshold deletes people, not defects.
for op in operators:
    z[op] = (features[op] - features[op].median()) / features[op].iqr()

score = w @ z          # composite, weights hand-set and stated

# 3. keep a quota per operator so no demonstrator is wiped out
keep = concat(top_k(score[op], quota[op]) for op in operators)
```

### Trimming — the part with a built-in validation

Two **independent** estimators of the task span:

- **from poses (free):** the interval where combined hand speed exceeds a threshold; trim idle
  head and tail.
- **from VIP (GPU):** the interval where the progress curve is monotonically rising.

If they agree, you have cross-validated a trim with no labels. If they disagree, that gap is a
finding — likely episodes where the hands move but the task isn't advancing, which is exactly
the "motion without progress" failure a jerk metric alone cannot see.

**A trim is an index range, applied to everything at once.** `trim_start`/`trim_end` must slice
the pose arrays and the frame indices identically — the poses *are* the actions here, and a
manifest that trims frames 40–160 while leaving the pose array whole silently destroys the
observation↔action correspondence a policy trains on. Emit the span once, apply it everywhere,
and assert `len(poses[a:b]) == len(frames[a:b])` in the writer.

**Do not curate down to textbook successes.** DROID and most demonstration corpora are
pre-filtered to successes, which is precisely why failure data is scarce in this field. Quality
and success are different axes: a clean, well-tracked, well-lit recording of a *failed* cup
placement is high-quality data. Score them separately and keep the good recordings of failures —
say so explicitly in the report, because a curation engine that quietly deletes all failures is
a known way to make downstream policies worse.

---

## 4. Output — what you actually hand over

### `frames.parquet` — the evidence, ~920k rows

| column | type | note |
|---|---|---|
| `episode_hash` | str | |
| `frame_idx` | int | |
| `jpeg_bytes` | int | exact, from the shard index |
| `z_bytes` | float | z-scored within episode |
| `is_blur` | bool | `z_bytes < -2` |

### `episodes_scored.parquet` — one row per episode, all 5,409

`episode_hash · operator · num_frames · blur_frac · size_cv · jerk_rms · sal · idle_frac ·
progress_final · monotonicity · score · score_z_within_operator · trim_start · trim_end`

### `keep_drop.csv` — the deliverable

```csv
episode_hash,operator,decision,score,reason,trim_start,trim_end
692e9f8d3971,6871e269,keep,0.82,,14,158
692e98937641,690cb8f6,drop,0.04,too_short:7_frames,,
692eb303ea43,6903686e,keep,0.61,,9,91
692ea16696d2,6846f7cf,drop,0.19,blur_frac:0.31,,
```

Every drop carries a machine-readable reason. That is what makes it auditable rather than an
opaque score.

### The validation report

1. **Precision / recall** against ~150 hand labels, on **held-out operators** — not a random
   split, or near-duplicate clips from the same person land on both sides and the number lies.
2. **Operator fairness table** — keep-rate per operator. If it's 80% for some and 25% for
   others, your scorer is a person detector and you say so.
3. **Ablation chart** — held-out action-prediction error of a small proxy policy trained on
   *random-N* vs *quality-N* vs *quality+diversity-N*. This is the "validation report using
   proxy metrics" the brief asks for.
4. **~15 preview MP4s** of the frames your score calls worst, as evidence the number means
   something.

---

## 5. Build order

| # | Step | Needs | Output |
|---|---|---|---|
| 1 | ~~shard-index scanner, local~~ | — | folded into step 2 |
| 2 | ✅ **done** — `curate.py` on Modal, all 5,409 | Secret | `frames.parquet` |
| 3 | pose kinematics, Modal | — | tier-2 features |
| 4 | score + per-operator normalise + `keep_drop.csv` | — | **a complete track-1 answer, no GPU** |
| 5 | hand-label 150 on previews | 30 min of your time | validation numbers |
| 6 | VIP progress on survivors | GPU | completion + trim |
| 7 | proxy-policy ablation | — | the money chart |

**Step 4 is the checkpoint.** Everything up to it is CPU-only and cheap, and it already
constitutes a defensible curation engine. Steps 5–7 are what make it credible rather than
merely plausible.

### Tier 1 results — measured, 2026-08-15

`modal run curate.py --limit 0`: **5,407 / 5,409 episodes, 904,511 frames, 66 seconds.**
Two episodes have no image shard at all (`processed_v3/mecka/flagship/692e99e9ba04…` and
`…692ea8fb727c…`) — immediate drops, invisible from SQL. 35 episodes disagree with SQL's
`num_frames`.

**The confounder check, and it is decisive.** Fraction of each feature's variance explained by
`operator` (η²):

| feature | η² | verdict |
|---|---|---|
| `median_bytes` | **84.3%** | ✗ unusable — pure operator/rig signature |
| `n_frames` | **70.5%** | ✗ mostly *who*, not *how well* |
| `longest_low_run` | 44.2% | ⚠ needs per-operator normalisation |
| `blur_frac` | 27.6% | ✓ mostly real |
| `low_frac` | 23.1% | ✓ mostly real |
| `size_cv` | 20.2% | ✓ mostly real |

This confirms the core design choice: **absolute JPEG size is worthless as a quality signal** —
it encodes the operator's rig and lighting, not their technique. Only the *within-episode
z-scored shape* of the byte series carries transferable signal. Mean `blur_frac` still varies
10× across operators (0.008 → 0.081), so per-operator normalisation stays mandatory.

**Hard-drop candidates** (thresholds provisional, pending the label set):

```
no image shard                        2   (0.0%)
n_frames < 60                         3   (0.1%)
blur_frac > 0.25                      1   (0.0%)
longest_low_run > 30                331   (6.1%)
size_cv > 0.15                      215   (4.0%)
UNION                               497   (9.2%)
```

Caveat to carry forward: `longest_faint_run` drives most of that 9.2% and is the most
operator-confounded of the surviving features. Operator `670be676…` holds 4 of the 8 worst
episodes despite owning only 51 of 5,407 — that is a session or rig fault, not 4 bad
demonstrations, and it must not be scored as if it were.

### Tier 2 results — poses, 5,407 episodes, ~7 min

`modal run curate.py::poses --limit 0` — 41 kinematic features per episode from `left/right
.obs_ee_pose`, `obs_head_pose`, `*.obs_keypoints`. Four things the data says:

**1. `hand_span_mean` is 84.3% operator — it is literally hand size.** An anthropometric ID
badge, not a quality measure. Ranked alongside `median_bytes` (84.3%) and `duration_s` (70.5%)
as features that must never enter a quality score.

**2. Trimming matters for a tail, not the median — and the two estimators fail at opposite ends.**

*(First attempt was degenerate: thresholding at `percentile(speed, 25)` puts 75% of frames above
the line by construction, so every span came out ≈[0,1] and η² looked near-zero only because the
feature had no variance. Re-run with a threshold at 15% of each episode's own peak speed.)*

```
median dead time   2.5%  (0.14 s of 5.7 s)   <- these clips are already tight
>20% dead        14.9%
>40% dead         5.0%
act_start p95    35.6%   act_end min 39.1%
```

So trimming is worth ~15% of the slice, not all of it. η² is now **9.3–14.7%** for
`act_start`/`act_end`/`act_span` — genuinely low against `duration_s` at 70.5% and
`hand_span_mean` at 84.3%, and this time on a feature with real variance (σ = 0.128).

The per-frame cross-check against the independent byte-detail signal:

| zone | frames | mean z | faint rate |
|---|---|---|---|
| dead head | 54,248 | **+0.166** | 8.7% |
| active | 825,560 | −0.027 | 14.5% |
| dead tail | 24,703 | **−0.307** | 23.9% |

The dead tail has markedly *less* image detail — 1.65× the faint rate of the active zone —
confirming the pose-derived boundary from a completely independent measurement. But the dead
head has *more* detail than the active zone. That asymmetry is the finding: at the start the
scene is fully set up and static (high detail, no motion); at the end the hands have withdrawn
(low detail, no motion). **Motion energy is the only signal that can find the dead head; image
detail independently confirms the dead tail.** Neither alone trims both ends, which is why the
episode-level correlation between them is only +0.158 — they are complementary, not redundant.

**3. There is no dominant quality axis.** PCA over all features: PC1 explains **13%**, and six
components reach only 52%. Quality here is genuinely multi-dimensional. A single scalar score
will discard most of the variance — report a vector, or at minimum say which axis a score is on.

**4. ★ Visual and kinematic quality are uncorrelated.**

```
corr(low_detail_frac,   head_jerk_rms)  = +0.024
corr(low_detail_frac,   right_jerk_rms) = +0.012
corr(low_detail_frac,   head_speed_mean)= -0.090
corr(faint_frac,        head_speed_max) = +0.022
```

Near zero, all of them. These are two genuinely independent views of the same episode, so a
curation score needs both — filtering on one tells you nothing about the other. This is the
single most useful structural fact from the run.

### ★ Tier 3 results — DINOv2 on 300 episodes: the visual space is an operator space

`modal run curate.py::embed --sample 300` — 16 frames per episode sampled across the *active*
span, range-fetched via the shard index, DINOv2-base CLS embeddings on an L4. 4,800 frames,
**2m26s**, 300/300 succeeded.

**The headline: a general-purpose vision model does not see the task here. It sees the
workstation, and the workstation is the operator.**

```
chance nearest-neighbour-shares-operator          9.2%
raw DINOv2                                       98.0%     <-- 11x chance
mean cosine, same operator                        0.755
mean cosine, different operator                   0.474
near-duplicate pairs > 0.95                    340, of which 100% same-operator
```

Every standard de-confounding trick fails:

| representation | NN shares operator | same/diff separation | cluster purity |
|---|---|---|---|
| raw DINOv2 | **98.0%** | +0.283 | 0.72 |
| per-operator mean-centred | **89.8%** | −0.025 | 0.47 |
| minus top-1 PC | 98.3% | +0.451 | 0.62 |
| minus top-5 PCs | 98.3% | +0.189 | 0.46 |
| minus top-20 PCs | 95.9% | +0.017 | 0.28 |

Per-operator centring drives the *global* separation to zero (−0.025) yet leaves the nearest
neighbour on the same operator 89.8% of the time. Removing up to 20 global principal components
barely moves it. Operator identity is not a low-dimensional nuisance direction that can be
projected out — it is distributed through the whole representation.

The reason is structural, not a model failure: `cup_on_saucer` is 5,409 clips of *one* task with
*one* object in a *fixed* workstation. There is essentially no task variance to see, so the only
thing left for a vision model to encode is the room.

### The consequence — and it splits cleanly

| comparison | operator confound | verdict |
|---|---|---|
| **across** episodes (diversity scores, coverage selection, dedup) | fatal | ✗ measures the camera |
| **within** an episode (progress, trimming, retry detection) | cancels — operator is constant | ✓ clean |

So for this slice, embedding-geometry diversity selection is dead: it would rank operator
diversity, which `app.episodes` already gives you for free from the `operator` column. What
survives is that **cross-episode dedup still finds real within-operator retakes** — 340 pairs
above 0.95 among just 300 episodes — and every *within-episode* use of the embeddings is
untouched by the confound.

The within-episode progress signal is real. Mean cosine similarity of each frame to the final
frame, across all 300 episodes:

```
0.681 0.682 0.679 0.673 0.669 0.665 0.660 0.661 0.672 0.696 0.734 0.760 0.771 0.784 0.821 1.000
        <------------- diverging ------------->        <------------- converging ------------->
```

A clean U — the scene starts resembling its own end state (cup and saucer laid out, static),
diverges as hands enter and occlude, then converges on the goal. The dip *is* the manipulation.
7.0% of episodes never dip: frame 0 is already >0.9 similar to the final frame, meaning nothing
visibly changed — a genuine "did anything happen?" flag that no pose statistic produced.

**Testable prediction:** run the same experiment on a multi-scene slice and the 98% should
collapse. If it doesn't, the confound is the rig, not the room.

### ★ Prediction confirmed — the confound was the slice, not the method

`multiscene.parquet`: 14,742 mecka episodes spanning 75 scenes and 83 tasks, sampled to 400
(58 scenes, 78 tasks, 315 operators). 32 frames each, motion-weighted. 8m47s on an L4.

```
cup_on_saucer (1 scene, 16f uniform):   NN shares operator  97.3%
multiscene    (58 scenes, 32f motion):  NN shares operator  25.0%
```

The decisive test is not nearest-neighbour rates (chance shifts with operator count) but the
mean cosine decomposed by pair type — does same-task-different-room still look alike?

| representation | same task<br>same scene | same task<br>**diff scene** | diff task<br>same scene | diff task<br>diff scene | **task effect** | scene effect | task/scene |
|---|---|---|---|---|---|---|---|
| raw | 0.650 | 0.530 | 0.342 | 0.206 | **+0.324** | +0.136 | **2.38** |
| per-episode centred | 0.002 | 0.001 | −0.001 | 0.000 | +0.000 | −0.001 | 0.59 |
| frame-delta | 0.057 | 0.036 | −0.000 | 0.000 | +0.036 | −0.000 | **91.3** |

*task effect* = (same task, different scene) − (different task, different scene).

**DINOv2 does see the task.** When scenes vary, the task effect is **2.4× the scene effect**.
The cup_on_saucer result was one task in one room — close to the worst case for this method,
and not a property of embeddings in general.

### Like-for-like: better sampling does *not* rescue a single-scene slice

Re-ran `cup_on_saucer` with the improved sampler so the comparison is controlled:

| sampling / representation | NN shares operator | lift |
|---|---|---|
| 16 frames, uniform | 97.3% | 11.0× |
| 32 frames, motion-weighted | **98.0%** | 11.6× |
| + frame-delta | 85.4% | 10.5× |
| + per-episode centred | 24.3% | 3.0× |

More frames and motion-weighting changed nothing — it got marginally *worse*. Frame-delta, which
achieved a 91:1 task/scene ratio on the multi-scene slice, only moves this from 97% to 85%. Only
per-episode centring breaks it, and the multi-scene test already showed that centring destroys
the task signal along with the room.

**Taken together with the multi-scene result, this is the clean statement:** the confound is not
an artefact of the representation, it is the information content of the data. One task, one
room, one object leaves nothing for a vision model to encode *except* the room. No amount of
sampling or projection recovers a signal that was never captured. When scenes do vary, the same
model and the same pipeline find the task at 2.4× the room.

### The progress curve reproduces across sampling schemes

32 motion-weighted frames, 350 episodes — mean similarity to the final frame:

```
0.731 0.728 0.718 0.705 0.690 0.735 0.807 0.872 0.933
      <-- diverging -->   ^dip    <-- converging -->
```

Mean dip depth −0.132, minimum at the midpoint. And the "nothing visibly happened" rate is
**7.7%** here against **7.0%** from the earlier 16-frame uniform run on a different 300-episode
sample. Two sampling schemes, two samples, same answer — that one is real.

### Data prep — three things tested

**1. Motion-weighted frame sampling (adopted).** Instead of frames evenly spaced in *time*,
space them evenly in *hand travel*: build the cumulative distribution of per-frame hand speed
and sample at equal quantiles, so frames concentrate where something is happening. Pose arrays
cost 0.25 MB, so this is nearly free. Also raised 16 → 32 frames per episode.

**2. Frame-delta representation (the big win).** Embed each frame, then use the *differences*
between consecutive frame embeddings rather than the embeddings themselves. Result:
**task effect +0.036, scene effect −0.000 — a 91:1 ratio.** Essentially pure activity signal
with the room removed. The catch: the absolute effect is ~9× smaller than raw, so it needs more
episodes for the same statistical power. Use raw for retrieval, frame-delta for any comparison
where the room is a confound.

**3. Per-episode centring (rejected).** Subtracting the episode's own mean frame embedding
removes the room *and* the task — every effect collapses to zero. Too aggressive.

### Edge detection does not isolate the hands — measured

Sobel/Canny over a real mecka episode:

```
edge pixels per frame                        27.6% of image
edges stable across >80% of frames            0.1% of image
of all edge pixels, fraction static              0%
```

`data/isolate_demo.png` shows why: the granite countertop generates far more edge response than
the hands do. Edges preserve static structure, which *is* the confound — so edge detection would
if anything amplify it.

The root cause defeats every fixed-camera technique: **the camera is head-mounted and travels
31.8 cm per episode** (1.88 mm/frame).

| technique | result |
|---|---|
| temporal-median background subtraction | 43.2% of the image "differs" — ghosting, not foreground |
| frame differencing | corner (background) 5.21 vs centre (hands) 10.59 — only **2.0×** |
| static edge masking | nothing is stable in pixel coordinates |

**What should work instead: crop to the hands using the keypoints.** `intrinsics` is mandatory
and present (`K` is 3×4), and `*.obs_keypoints` is dense — 21 landmarks per hand, every frame,
zero dropout measured. Projecting and cropping would remove the room entirely and is robust to
camera motion because the crop follows the hands.

**Solved.** The transform is in `act.py::ActionChunkCoordinateFrameTransform` —
`target_se3.inverse() @ chunk_se3`, then `pose_utils.cam_frame_to_cam_pixels`. Quaternions are
stored `[qw,qx,qy,qz]`, so scipy needs `[4,5,6,3]` reordering:

```python
head_se3 = np.eye(4)
head_se3[:3, :3] = Rotation.from_quat(head_pose[[4, 5, 6, 3]]).as_matrix()
head_se3[:3, 3]  = head_pose[:3]
cam    = (np.linalg.inv(head_se3) @ homogeneous_keypoints.T).T     # world -> camera
pixels = (intrinsics @ cam.T); pixels = (pixels / pixels[2]).T[:, :2]
```

Verified visually in `data/keypoint_projection.png` — MANO skeletons land on the hands across
the episode. My earlier failure was a bad *metric*, not bad maths: I scored candidates by
"fraction landing on foreground", but on a moving camera 36% of the image is ghosting, so the
metric could not distinguish them. **The correct transform was among the ones I tried; I just
rendered the wrong candidate.**

### ★ Hand-cropping does not fix the confound — it concentrates it

12,733/12,733 frames cropped to the projected hand box (margin 1.8× so the manipulated object is
included), 400 cup_on_saucer episodes, 6m45s.

| condition | NN shares operator | lift |
|---|---|---|
| 16f uniform, full frame | 97.3% | 11.0× |
| 32f motion-weighted, full frame | 98.0% | 11.6× |
| **32f motion-weighted, hand-cropped** | **97.8%** | 11.5× |
| + frame-delta on crops | 85.7% | 10.5× |

No effect whatsoever. In hindsight this is obvious and we had already measured it: **hand span
is 84.3% operator** — the hands *are* the person. Their size, skin tone, sleeves, and the way
they hold the cup identify the demonstrator at least as strongly as the room does. Cropping to
the hands crops to the most operator-specific object in the frame.

Cropping actually made episodes look *more* alike, not less — median pairwise cosine rose
0.485 → 0.538 and near-duplicate pairs rose 569 → 608 — because it removed scene variety while
keeping the person.

**But it sharpens the within-episode signal**, exactly as the dividing line predicts:

| | full frame | hand-cropped |
|---|---|---|
| progress dip depth | −0.132 | **−0.174** |
| "nothing happened" rate | 7.7% | 4.9% |

A 32% deeper dip: focusing on the hands makes the manipulation more visible and reclassifies
some apparent non-events as real ones. Use hand-cropping for progress and trimming; it buys
nothing for cross-episode comparison.

### Correction: `blur_frac` was the wrong name

It is now `low_detail_frac` / `faint_frac` / `longest_faint_run`. The +0.95 correlation with
Laplacian variance is real, but *both* metrics measure image detail, and detail ≠ sharpness when
the scene changes. Two pieces of evidence say this is content, not blur:

- **lag-1 autocorrelation of the byte series is +0.615** — smooth drift. Motion blur is bursty
  and would show near-zero autocorrelation.
- **low frames concentrate 2× in the final 10% of episodes** (20.0% of them, against 10% by
  chance), where mean z drops to −0.41. Everywhere else the distribution is close to uniform.

That pattern is the hands and cup leaving the frame as the demonstrator withdraws. Which makes
it a **trimming signal, not a quality signal** — and a second, independent estimator of
`act_end` to cross-check against the pose-derived one.

---

## 5b. Threshold, encoders, and failure taxonomy

### The proxy-policy ablation cannot set a threshold — and that is the finding

`ablate.py`: ridge on pose → future displacement, evaluation set fixed and split **by operator**,
5 seeds per point.

*First attempt was saturated.* Next-frame prediction gave identical held-out MSE at 313 and
3,131 episodes. Not "curation doesn't help" — a 15-parameter model learns "hands move smoothly"
from 300 episodes and cannot use more.

*Fixing it barely helped.* Longer horizons (0.5 s, 1.0 s) and 1024 random Fourier features
finally made data quantity matter, but only by **1.3%–2.6%** between 10% and 100% of the data.
Quality ranking captured ~1% of that, at 1.2σ. Not significant at any horizon or keep fraction.

*And there is no natural break in the score.* Unimodal — one histogram maximum, knee at 92%
(i.e. no knee). Calibration against the 63 independently-verifiable bad episodes is weak:

```
median percentile of a known-bad episode:   37.3%
median percentile of everything else:       50.1%
cutting the bottom 10% catches 27% of known-bads, at a cost of 541 episodes
```

**Conclusion: no threshold is intrinsic to this data.** The defensible cuts are (a) the 63 hard
drops, (b) the 257 episodes (4.8%) failing *both* independent views — 28% above the 201 expected
by chance — and (c) any percentile, explicitly labelled a budget decision.

### A stronger encoder does not help — measured

Same 400 episodes, 32 hand-cropped motion-weighted frames:

| encoder | params | dim | NN-same-op | lift | dip depth | never-dip | dup > 0.95 |
|---|---|---|---|---|---|---|---|
| dinov2-base | 86M | 768 | 97.8% | 11.5× | −0.174 | 4.9% | 608 |
| dinov2-large | 300M | 1024 | **98.0%** | 11.6× | −0.186 | 6.3% | 898 |
| siglip2 | 375M | 768 | 95.2% | 11.2× | **−0.195** | **2.3%** | 1074 |

*(chance = 8.5%)*

A 4.4× parameter range moves the confound from 98.0% to 95.2% — still ~11× chance. **No encoder
escapes it.** But SigLIP2 is clearly best on the signal that *does* work: 12% deeper progress dip
than the baseline and less than half the never-dip rate. Better encoders also surface more
near-duplicates (608 → 1074).

**Recommendation: use SigLIP2**, not because it fixes cross-episode comparison — nothing does —
but because it is the strongest at within-episode progress, which is where the usable signal is.
Capacity is not the bottleneck; if you want a real step change, the axis is *modality* (a
video-native encoder such as V-JEPA 2 or InternVideo2), not size.

### Failure taxonomy — five independent axes

Before this, 96% of drops read `below_operator_quota`, which is not a reason. Five axes,
z-scored within demonstrator, max pairwise correlation **0.14**; k-means over the drop
population recovers exactly one cluster per axis, so these are distinct modes:

| axis | features | drops |
|---|---|---|
| `hesitant` | hand idle fractions | 211 |
| `poor_image` | low detail, faint frames, exposure instability | 196 |
| `jittery` | SPARC on both hands and head | 174 |
| `unsteady_camera` | head jerk, head idle | 151 |
| `incomplete` | short active span, low straightness | 112 |
| *two-axis combinations* | | 437 |
| `below_average` | **no identifiable defect** | 248 |

Three things this surfaces that the old manifest hid:

- **15.4% of ranked drops have no defect at all** — below their demonstrator's median, nothing
  more. That portion of the drop budget buys nothing.
- **24.4% of *kept* episodes carry a warning flag**, now exposed in a `flags` column so a
  downstream user can tighten without re-running anything.
- **Multi-axis failure is a confidence signal.** 437 dropped episodes fail two axes and 89 fail
  three or more, against 50 and 0 among kept episodes.

The manifest carries all five axis z-scores per row, so every decision is inspectable.

---

## 6. Evaluation of the generic video-curation architecture

The reference architecture (ingest → segment → cheap filter → GPU semantics → diversity →
manifest) is a sound **generic** pipeline, and its execution model is exactly right. But it is
written for raw teleoperation MP4s. EgoVerse ships processed Zarr **with proprioception**, and
four of its stages would have you re-derive from pixels what is already sitting in the arrays.

### Adopt

| Idea | Why it improves this plan |
|---|---|
| **`.spawn_map()`** | Fire 5,409 jobs without the initiating process collecting every result. Right for tiers 1–2, where results go to a Volume anyway |
| **Manifest by reference, never re-encode** | Already the design — worth stating as a hard rule. Never write 920k JPEGs |
| **Cheap-gates-expensive** | Already the tier structure. The reference states it more crisply than I did |
| **Observation↔action co-trimming** | ★ The best idea in it. Now §3 — a trim span must slice poses and frames identically |
| **Deliberately preserve failures** | ★ Now §3. Quality ≠ success, and success-only corpora are a known field-wide problem |
| **`uv_pip_install` in the image** | Faster, more reproducible builds than `pip_install` |
| **Sample 2–5 fps for curation, not 30** | Correct. 32 frames of 170 is ~5 fps and is what tier 3 does |

### Reject, with reasons

| Stage as written | Why it does not apply |
|---|---|
| **ffprobe metadata / validate files** | No video containers. Metadata is Postgres — and `zarr.json` is actively unreliable (`task_name: "debug"`, `fps` wrong by 2×). Validate against SQL |
| **Segment into 5–20 s candidate clips** | `cup_on_saucer` episodes are *already* atomic ~6 s clips. Segmentation is a no-op for this slice; it only matters for the Aria 1–6 min sessions. Here it collapses into head/tail trimming |
| **Optical flow / mean frame difference for motion** | ★ The biggest miss. `obs_ee_pose` gives real 3D hand kinematics at 30 fps. Estimating motion from pixels when you have proprioception is strictly worse *and* far more expensive |
| **Perceptual hash for duplicates** | Requires decoding every frame. Pose-trajectory similarity is cheaper and more meaningful; the per-frame JPEG byte series is a free episode signature from tier 1 |
| **OpenCV blur/exposure over all frames** | JPEG byte size correlates **+0.95** with Laplacian variance and comes free from the shard index. Decode only to validate the proxy, not to compute it |
| **"success/failure labels" as a GPU stage** | There are no success labels in EgoVerse and no off-the-shelf robot-success checkpoint. This is VIP progress plus your own hand labels, not a model call |
| **`CloudBucketMount` for the read path** | Fine for whole-file reads, but tier 1 needs *the last 16 KB of an 11.7 MB object*. `boto3` `Range=` expresses that exactly and provably transfers nothing else. Use the mount for weights or bulk, `boto3` for ranged reads |

### The omission that matters most

The reference plan lists `operator` once, as a diversity axis. In this dataset it is the
**primary confounder**: two of 22 operators produce 26% of the slice, and their median clip is
108 frames against 236 for others. Any global threshold deletes people rather than defects.
That is why §3 normalises per operator and §4 validates on held-out operators — neither appears
in the generic architecture, and without them the manifest is a person detector with a good
score attached.

Relatedly, the reference plan ends at a manifest and never asks whether the manifest is better
than a random subset of the same size. That question is the actual deliverable (§4, item 3).

---

## 7. Working in your Modal workspace

The CLI is installed (`modal 1.5.4`, via `uv tool install modal`) and Modal's own agent skill is
at `.claude/skills/modal/` — refresh it with `modal skills update`.

### One-time auth

```bash
modal setup                     # opens a browser, writes ~/.modal.toml
modal profile list              # which workspace am I in?
modal workspace current
```

If you belong to several workspaces (personal vs a hackathon team), `modal profile activate
<name>` switches. Check this before a long run — deploying to the wrong workspace is the classic
2 a.m. mistake.

### Push the R2 credentials in as a Secret

Never bake them into the image or commit them. Read them out of `~/.egoverse_env`:

```bash
set -a; . ~/.egoverse_env; set +a
modal secret create egoverse-r2 \
  R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  R2_ENDPOINT_URL="$R2_ENDPOINT_URL" \
  BUCKET=rldb
modal secret list
```

### The four ways to run code

| Command | What it does | Use it for |
|---|---|---|
| `modal run app.py` | runs `@app.local_entrypoint()`, streams logs, tears down | ★ normal iteration |
| `modal run --detach app.py` | same but survives your laptop closing | the 5,409-episode passes |
| `modal shell app.py::fn` | interactive shell **inside that function's container** | ★ debugging "does zstd exist in my image" |
| `modal deploy app.py` | persistent app, callable later / on schedule | demo-day endpoints |

`modal serve app.py` hot-reloads web endpoints — for a Gradio dashboard, not for batch work.

**`modal shell` is the one people forget.** It drops you into the exact image with the exact
mounts and secrets, so you can check paths and imports interactively instead of by
edit-run-wait. Use it the first time each tier runs.

### Watching and fetching results

```bash
modal app list                        # what's running
modal app logs <app-id>               # stream logs
modal volume ls egoverse-feat /       # what landed on the Volume
modal volume get egoverse-feat /frames.parquet ./frames.parquet
```

The Modal dashboard (`modal dashboard`) shows per-container timing, which is how you find out
that one container is doing 80% of the work because your `.map()` chunks are uneven.

### Skeleton for this project

```python
import modal

app = modal.App("egoverse-curator")

image = (
    modal.Image.debian_slim()
    .uv_pip_install("boto3", "numpy", "pandas", "pyarrow",
                    "zarr==3.1.5", "simplejpeg", "zstandard")
)
vol = modal.Volume.from_name("egoverse-feat", create_if_missing=True)
r2 = modal.Secret.from_name("egoverse-r2")


@app.function(image=image, secrets=[r2], volumes={"/feat": vol},
              max_containers=50, retries=modal.Retries(max_retries=3))
def scan_shard_index(row: dict) -> dict:
    ...                                       # 16 KB ranged read -> per-frame sizes
    np.save(f"/feat/{row['episode_hash']}.npy", sizes)   # big data -> Volume
    return {"hash": row["episode_hash"], "blur_frac": ...}   # scalars -> return


@app.local_entrypoint()
def main(limit: int = 200):
    rows = pd.read_parquet("episodes.parquet").head(limit).to_dict("records")
    results = list(scan_shard_index.map(rows, return_exceptions=True))
    ok = [r for r in results if not isinstance(r, Exception)]
    print(f"{len(ok)}/{len(rows)} ok")
```

Pin `zarr==3.1.5` in the image for the same reason as locally — newer versions reject these
stores. Keep `max_containers` capped so 5,409 concurrent range reads don't get you rate-limited
by R2.

