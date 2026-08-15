# Systems Design — EgoVerse × Modal

How the dataset actually reaches a model, what comes back, and what you can build on top.

Companion to `EGOVERSE_NOTES.md` (what the data is) and `MODAL_NOTES.md` (how Modal works).

---

## 0. The central observation

**All three tracks are the same system with a different head.**

Every track needs: enumerate episodes → turn each episode into a small fixed-size
vector → do arithmetic on the resulting matrix. Only the last step differs.

```
Track 1 (curation)  = argmax over subsets of a quality+coverage objective
Track 2 (diversity) = a scalar function of the matrix
Track 3 (reward)    = a classifier head on the same vectors
```

So build the shared pipeline once. Pick your track at stage 3, not stage 0. If you
finish early you get the other two nearly free — and the combined system (curate
*for* diversity, *filtered by* success) is a stronger demo than any single track.

---

## 1. Three taps into an episode

The single most important design decision: an episode is not one thing. It has three
layers with cost differences of ~5 orders of magnitude.

| Tap | What | Size/episode | Compute | Latency |
|---|---|---|---|---|
| **A. Metadata** | SQL row + `zarr.json` attrs | ~1 KB | none | ms |
| **B. Kinematics** | `obs_ee_pose (T,7)`, `obs_keypoints (T,63)` | ~0.5–2 MB | CPU | ~1 s |
| **C. Pixels** | `images.front_1` JPEG blobs | 100–800 MB | GPU | ~30 s |

Zarr's chunking means **you can read B without paying for C.** That's the whole
reason this is tractable in a hackathon. Design every stage as a filter that
shrinks what the next stage has to touch.

A rough shape for a 500-episode slice:

```
stage 1 (B only, 100 CPU containers)  ~3 min    $~0
stage 2 (C, 20 GPU containers)        ~15 min   $~5
stage 3 (numpy on a 500×1152 matrix)  ~0.5 s    $0     ← the entire rest of the project
```

Once stage 2 has run, your dataset is a ~1 MB float16 matrix that fits in L2 cache.
Every experiment after that is instant. **The GPU pass is a one-time compiler from
video into linear algebra.**

---

## 2. The pipeline

```
                    Postgres app.episodes
                            │
                            ▼
                    episodes.parquet          ← the work list. hash, path, task, lab, fps
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ╔═══════════════════╗       (skipped for dropped episodes)
    ║ STAGE 1 · CPU     ║
    ║ pose-only read    ║  CloudBucketMount → zarr sparse read
    ║ 100 containers    ║  → jerk, SAL, TED, idle %, path len, span coverage
    ╚═════════╤═════════╝
              │  ~20 float32 per episode
              ▼
        kinematics.parquet ────────────────┐
              │                            │
              ▼                            │
    ╔═══════════════════╗                  │
    ║ STAGE 2 · GPU     ║                  │
    ║ s5cmd bulk copy   ║  N=32 sampled frames                 
    ║ @modal.batched    ║  → SigLIP2/DINOv3 (N,1152)
    ║ 20 containers     ║  → VIP progress   (N,)
    ╚═════════╤═════════╝
              │  one .npz per episode on the Volume
              ▼
      /feat/<hash>.npz ────────────────────┤
                                           │
                                           ▼
                              ╔═══════════════════════╗
                              ║ STAGE 3 · CPU, numpy  ║
                              ║  E×D matrix in RAM    ║
                              ╚═══════╤═══════════════╝
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
             greedy selection     Vendi / DPP        logistic head
              (track 1)            (track 2)          (track 3)
                    └──────────────────┼──────────────────┘
                                       ▼
                          ╔════════════════════════╗
                          ║ STAGE 4 · proxy policy ║  ← the validation that makes it real
                          ║ tiny BC on cached embs ║
                          ╚════════╤═══════════════╝
                                   ▼
                        Gradio dashboard (modal deploy)
```

---

## 3. Stage 1 — kinematics, CPU only

### Getting the bytes

Two access modes, pick per stage:

```python
# Stage 1: sparse reads. Mount the bucket, let zarr pull only the chunks it needs.
r2 = modal.CloudBucketMount(
    "rldb",
    bucket_endpoint_url=os.environ["R2_ENDPOINT_URL"],
    secret=modal.Secret.from_name("egoverse-r2"),
)
# NOTE: CloudBucketMount wants AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the
# secret. setup_secret.sh writes R2_* names — alias both into the Modal Secret.

# Stage 2: dense reads. FUSE is slow over thousands of tiny JPEG chunks; bulk copy.
subprocess.run(["s5cmd", "--endpoint-url", ep, "cp",
                f"s3://rldb/processed_v3/{prefix}/{h}.zarr/*", f"/tmp/{h}.zarr/"])
```

### What you compute

```python
def kinematics(pose: np.ndarray, fps: float) -> dict:      # pose: (T,7)
    t = pose[:, :3]                                        # translation, meters
    dt = 1.0 / fps                                         # ← NOT optional, see gotchas
    vel = np.diff(t, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return {
        "jerk_rms":   float(np.sqrt((jerk ** 2).sum(-1).mean())),
        "path_len":   float(np.linalg.norm(np.diff(t, axis=0), axis=-1).sum()),
        "sal":        spectral_arc_length(vel, fps),
        "idle_frac":  float((np.linalg.norm(vel, axis=-1) < 0.01).mean()),
        "duration_s": len(pose) / fps,
    }
```

Plus, from metadata: annotation count, fraction of frames covered by a span, number
of distinct sub-steps, bimanual overlap fraction.

Returns ~20 floats. The whole dataset's kinematics is a parquet file you can email.

### Four gotchas that will silently ruin this

1. **Normalize by `fps`.** It's 30 *or* 60 depending on capture device. Raw
   frame-difference jerk measures the camera's frame rate, not the human's motion.
2. **Work in the head-centric frame.** `obs_head_pose` is world-frame SLAM. Raw
   coordinates encode *which room the demo happened in.* Re-express EE pose relative
   to head pose (the repo's own training convention) or your metrics cluster by
   scene instead of by behavior.
3. **Quaternions are sign-ambiguous** (`q ≡ -q`). Never finite-difference them
   directly. Use geodesic angular distance: `2 * arccos(|<q_t, q_{t+1}>|)`.
4. **`lab` is a confounder.** Vendor/device determines resolution, lighting, and
   noise floor. Any metric you compute will partly measure vendor. Always report
   your score both raw and stratified by `lab` — see §7.

---

## 4. Stage 2 — pixels into vectors

### Frame → tensor

```python
blob   = z["images.front_1"][i]           # varlen bytes, one JPEG
img    = simplejpeg.decode_jpeg(blob)     # (H, W, 3) uint8 — no ffmpeg, no codec
batch  = preprocess(imgs)                 # (B, 3, 384, 384) float16
emb    = model.get_image_features(batch)  # (B, 1152)
```

### Which frames

Not all of them. `T` is 1,000–5,000; you need ~32.

- 16 uniform over the episode — the backbone of the representation
- + the frames at every annotation `start_idx` / `end_idx` — **sub-task boundaries
  are the semantically dense moments.** This is free information the dataset hands
  you and most people will ignore it.
- + first and last frame — track 3 cares disproportionately about the terminal state

### What you emit

One `.npz` per episode on the Volume, keyed by hash:

```python
np.savez(f"/feat/{h}.npz",
         emb_mean     = emb.mean(0),        # (1152,)  episode-level vector
         emb_frames   = emb,                # (32,1152) temporal detail
         frame_idx    = idx,                # (32,)    which frames these were
         vip_progress = progress)           # (32,)    track 3's raw signal
```

Return `{"hash": h, "ok": True}` and nothing else. **Never return the array** —
anything >2 MiB round-trips through us-east and destroys your throughput.

### Why VIP in the same pass

VIP (`facebookresearch/vip`) is a ResNet-50 trained self-supervised on Ego4D human
video — the closest available domain match to EgoVerse, and it needs no action
labels. Distance from each frame's embedding to the *final* frame's embedding is a
dense progress signal. It's ~2 GB and runs on frames you've already decoded, so the
marginal cost of adding it to the featurizer is close to zero. Do it even if you're
not doing track 3; a progress curve is a good curation feature too.

---

## 5. Stage 3 — the heads

Everything below operates on `X: (E, 1152)` and `K: (E, 20)` in memory. No Modal
required, though wrapping it in a `gpu=None` function keeps the demo one-click.

### Track 1 — Curation Engine

Selection is a submodular maximization. Greedy facility-location with a quality prior:

```python
def select(X, quality, budget):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = X @ X.T
    chosen, cover = [], np.full(len(X), -1.0)
    for _ in range(budget):
        gain = np.maximum(sim, cover[:, None]).sum(0) - cover.sum()   # marginal coverage
        gain[chosen] = -np.inf
        pick = int(np.argmax(gain + LAMBDA * quality))
        chosen.append(pick)
        cover = np.maximum(cover, sim[pick])
    return chosen
```

`quality` comes from stage 1 (low jerk, low idle, plausible duration, annotation
coverage). Greedy gets you the `(1 − 1/e)` guarantee, which is a nice thing to say
out loud to a judge. Runs in ~0.2 s for E=500.

**Deliverable:** `keep_drop.csv` — hash, keep/drop, quality score, marginal
coverage, and the reason it was dropped (redundant-to-X / jerky / idle / failed).
Per-episode *reasons* is what makes this feel like an engine rather than a threshold.

### Track 2 — Diversity

Vendi Score is the headline number: `exp(H(eigenvalues of the normalized similarity
kernel))`. It's the effective number of distinct episodes. Needs no reference set,
is a single scalar, ranks two subsets directly, and is ~30 lines of numpy.

**The differentiator: diversity is not one number, it's a vector.**

You have three independent views of every episode and they disagree:

| View | Source | "Diverse" means |
|---|---|---|
| **Visual** | SigLIP2/DINOv3 embedding | different scenes, objects, lighting |
| **Kinematic** | stage-1 feature vector | different motion strategies |
| **Semantic** | annotation spans | different sub-task decompositions |

Two subsets can have identical visual Vendi and 3× different kinematic Vendi. A set
of 200 clothes-folding demos in 200 rooms is visually diverse and kinematically
degenerate — and it's the kinematic axis the policy actually learns from. **Report
`(V_visual, V_kinematic, V_semantic)` and show a case where they disagree.** That
single chart is the whole track.

### Track 3 — Human Reward Model

Don't prompt a VLM. Build features from the VIP progress curve and fit a tiny head:

```python
def progress_features(p):                  # p: (32,) VIP progress
    return [p[-1], p.max(), p[-1] - p.max(),           # final, peak, drawdown
            np.mean(np.diff(p) > 0),                   # monotonicity
            longest_plateau(p), np.polyfit(np.arange(len(p)), p, 1)[0]]
```

Concatenate with kinematics, hand-label ~200 episodes (one hour with a frame-grid
viewer), fit logistic regression. Trains in milliseconds, is fully interpretable,
and beats prompt-tuning a 7B VLM under time pressure.

**The three deliverables fall out directly:**
- *tagged episodes* → `predict()` over the slice
- *prevalence audit* → histogram of predictions grouped by task / lab / embodiment
- *confidence meter over a segment* → the VIP curve itself, plotted under the video

**Say the caveat before a judge does:** demo datasets are success-biased, so your
positive class dominates. Report balanced accuracy and precision@k on the *failure*
class, not raw accuracy — a classifier that always says "success" will score 90%.

---

## 6. Stage 4 — the proxy policy (this is what makes track 1 real)

"Validation report using proxy metrics" is where most curation projects wave their
hands. You can do it properly, because the action space here is *tiny*.

Train a behavior-cloning model that predicts the next K end-effector poses from the
current frame embedding and current pose:

```
input :  emb_frames[t] (1152,)  ⊕  pose[t] (7,)     → both already cached
output:  pose[t+1 : t+K] deltas (K,7)
model :  2-layer MLP, ~2M params
```

Because embeddings are precomputed, **training is a matmul over cached vectors** —
seconds per run, not hours. You can afford 20 ablations.

The money chart:

```
held-out
action MSE │ ▲
           │  ╲___                    random subset
           │      ╲──────
           │   ╲                      quality-only
           │    ╲___
           │     ╲    ╲___            quality + coverage  ← yours
           └──────────────────────▶
             25%   50%   75%  100%  fraction of data kept
```

If your curated 50% beats random 100%, you have the slide the whole track is asking
for. This is exactly the paper's own claim — *alignment beats scale* — reproduced
with your selector.

**State the limitation honestly:** held-out action MSE is an open-loop proxy for
closed-loop success. It is the standard proxy and it's what fits the budget. Saying
this yourself reads as rigor; being asked it reads as a gap.

---

## 7. The honesty check that wins arguments

Run this and put it on a slide regardless of track:

> Is my metric measuring the *task*, or the *camera*?

Compute your score on a set stratified by `lab` (same task, different vendor) and on
a set stratified by `task` (same vendor, different task). If vendor variance
dominates task variance, your embedding is measuring capture hardware. The fix is
cheap: mean-center embeddings per `lab` before scoring.

Nobody else will run this check. It costs 10 minutes and it's the difference between
"we computed a number" and "we validated a number."

---

## 8. Build order

Strictly sequential — each step is demoable on its own, so you're never holding a
half-finished system when time runs out.

| # | Step | Time | Unlocks |
|---|---|---|---|
| 1 | `setup_secret.sh` → `modal secret create` | 20 min | everything |
| 2 | Open **one** episode, print real keys and shapes | 30 min | validates §2 of the notes |
| 3 | `episodes.parquet` from Postgres for one filter preset | 30 min | the work list |
| 4 | Stage 1 `.map()` over the slice → `kinematics.parquet` | 1 hr | **a complete track-1 baseline, no GPU** |
| 5 | Stage 2 featurizer → feature Volume | 2 hr | tracks 2 and 3 |
| 6 | Stage 3 head for your chosen track | 1 hr | the deliverable |
| 7 | Stage 4 proxy policy | 2 hr | the validation chart |
| 8 | Gradio dashboard, `modal deploy` | 1 hr | the demo |

Step 4 is the checkpoint: if you have kinematics on a few hundred episodes and
nothing else, you still have a keep/drop recommendation with a defensible rationale.
Everything after that is upside.

Scope to **one filter preset** (`aria-fold-clothes`). One task held constant across
many demonstrators is the ideal setting for both diversity comparison and
success/failure labeling — the variation you measure is behavioral, not task-driven.
