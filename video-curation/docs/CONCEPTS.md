# Concepts — what actually happens to the data

Plain-language companion to `SYSTEM_DESIGN.md`. Read this one first.

---

## 0. Three corrections up front

**Modal has no models.** Modal is not an AI provider. It rents you computers that boot
in seconds and stop billing when idle. That's the whole product.

| Thing | What it is | Where it comes from |
|---|---|---|
| **Modal** | rented machines + a way to run your function on 100 at once | Modal, per second |
| **SigLIP2, DINOv3, VIP** | files of frozen numbers (pretrained weights) | free from HuggingFace |
| **Scorer, selector, classifier** | ~200 lines of numpy | you |

**The model never sees Zarr.** Zarr is a filing cabinet, not something a model reads.

**The model runs exactly once, in the middle, and it is dumb.** It converts pictures
into numbers. It does not curate, judge, or decide. All three tracks are what you do
*after* it, and none of them touch it again.

```
Zarr  →  decode  →  ┌───────┐  →  vectors  →  YOUR CODE  →  deliverable
                    │ MODEL │
    you             └───────┘         you
```

---

## 1. The one idea

**An image model turns a picture into a list of numbers, and similar pictures get
similar lists.**

The list is a location. 1152 numbers are coordinates in a 1152-dimensional space —
unpicturable, but the math works exactly like a map. Two photos of shirt-folding land
near each other; a kitchen lands far away.

Nobody programmed that. It emerged from training on hundreds of millions of
image–caption pairs with one objective: *pull each image toward its own caption, push
it away from everyone else's.* Do that long enough and the geometry ends up encoding
meaning.

**The reframe that matters: you are not asking the model questions. You are using it
as a measuring instrument** — a ruler that converts pixels into coordinates. Once
every episode has coordinates you are done with AI. Everything after is distance,
volume, and clustering.

One consequence: **individual numbers mean nothing.** Dimension 47 is not
"shirt-ness." Only relationships *between* vectors carry information. Never try to
interpret a single coordinate.

### What the network does mechanically

1. Chop the image into a grid of patches — 384×384 pixels becomes 27×27 patches of 14px.
2. Turn each patch into a vector. At this point it's a fancy pixel summary.
3. **Let every patch look at every other patch and update itself.** This is attention,
   and it's the step that makes it work — a patch of red fabric updates differently
   with a hand beside it than with a couch beside it. Context changes representation.
4. Repeat step 3 about two dozen times.
5. Average the patches into one vector.

A stack of "look around and revise" operations, ending in a summary of what's going on.

---

## 2. Zarr is a filing cabinet

A Zarr store is a directory: a JSON index plus thousands of small compressed chunk
files. Ask for frame 500 and Zarr consults the index, works out which chunk holds it,
downloads **only that file**, decompresses, and hands you raw bytes.

Think of a spreadsheet saved as thousands of tiny files — one per rectangle of cells —
plus a map of which file holds which cells. You can open three cells without loading
the sheet.

**This is the reason the project fits in a hackathon.** `obs_ee_pose` and
`images.front_1` are separate subdirectories. Reading poses touches ~1 MB. Reading
video touches ~340 MB. You can take the cheap one and skip the expensive one, per
episode, for free.

---

## 3. One episode, end to end

### What you start with

```
s3://rldb/processed_v3/aria/2026-03-14-10-22-51-330192.zarr/
├── zarr.json                 400 B    metadata
├── images.front_1/           340 MB   2,847 JPEGs
├── obs_head_pose/            160 KB   (2847, 7) float64
├── right.obs_ee_pose/        160 KB   (2847, 7) float64
├── right.obs_keypoints/      1.4 MB   (2847, 63) float64
└── annotations/              400 B    4 JSON blobs
```

```json
{"embodiment": "human_bimanual", "fps": 30, "total_frames": 2847,
 "task_description": "fold the shirt and place it on the stack", "lab": "aria"}
```

2,847 frames ÷ 30 fps = **95 seconds of someone folding a shirt.** 342 MB.

### Step 1 — poses in, numbers out (no model)

**IN** — two rows of `right.obs_ee_pose`:

```
frame 0:  [ 0.412, -0.238,  0.891,   0.997, 0.012, -0.071, 0.028]
frame 1:  [ 0.415, -0.236,  0.889,   0.996, 0.014, -0.073, 0.029]
           └── position (m) ──┘      └──── rotation ─────┘
```

**PROCESS** — subtraction. The hand moved 3.7 mm in 1/30 s → 0.11 m/s. Subtract
velocities for acceleration, again for jerk. Four lines of numpy.

**OUT** — one row:

```
hash                            jerk_rms  path_len  idle_frac  duration
2026-03-14-10-22-51-330192        41.2      8.71      0.22       94.9
```

**342 MB → 20 numbers, no GPU, ~1 second.** Already a quality signal: the hand
travelled 8.7 m and sat still 22% of the time.

### Step 2 — images in, vectors out (the model)

**IN** — not 2,847 frames. Sample 32:

```python
blob = z["images.front_1"][94]          # bytes, 78 KB, a JPEG
img  = simplejpeg.decode_jpeg(blob)     # (720, 1280, 3) uint8   ← pixels, 0-255
img  = resize_normalize(img)            # (3, 384, 384) float16  ← rescaled to ~-1..1
```

Stacked: `(32, 3, 384, 384)` — 14 million numbers.

**OUT** — `(32, 1152)`:

```
emb_frames[0]  = [ 0.0121, -0.0384,  0.0092, ..., -0.0117,  0.0417]
emb_frames[1]  = [ 0.0118, -0.0379,  0.0101, ..., -0.0122,  0.0409]
                   └────────────── 1152 numbers ──────────────┘
```

Frames 0 and 1 are nearly identical because the scene barely changed.

**14M numbers → 36,864 numbers.**

### What's now on disk

```
/feat/2026-03-14-10-22-51-330192.npz          148 KB
   emb_mean      (1152,)     ← the episode as one point
   emb_frames    (32, 1152)  ← the episode as a path through space
   vip_progress  (32,)       ← [0.02, 0.09, 0.21, ... 0.88, 0.91]
```

**342 MB → 148 KB.** 2,300× smaller and now a shape you can do algebra on.

Over 500 episodes, the entire dataset becomes:

```
X  (500, 1152)  float16   1.2 MB     ← visual
K  (500, 20)    float32    40 KB     ← kinematic
```

170 GB of video → 1.2 MB. It fits in cache. Everything from here is instant.

---

## 4. From 32 image vectors to one episode vector

A hidden step with real consequences. The model gives you one coordinate per *frame*;
an episode isn't a frame, so you have to collapse.

- **Mean-pool** → `(1152,)`. Cheap and robust. Answers "what stuff and what scene are
  in this episode." **Throws away time completely** — a successful fold and the same
  fold played backwards give identical vectors.
- **Keep all 32** → retains temporal structure, but episodes are no longer single points.
- **Both.** It's 64 KB per episode. Not worth agonising over.

The consequence: mean-pooled space answers *"are these two episodes about the same
thing?"* It cannot answer *"did this one go well?"* That's not a model flaw — it's a
fact about what a bag of frames contains.

---

## 5. Everything comes from one matrix

Normalise, then:

```python
S = X @ X.T          # (500, 500) — every episode against every other
```

One matmul, microseconds. Almost every operation below is a different way of reading
this one table.

### Find things

**Nearest neighbours** — sort any row of `S`. Run this first, always. If an episode's
top neighbours don't visibly resemble it, the embedding is broken and everything
downstream is noise. Also the best live demo: a judge points at an episode, you show
its twins instantly.

**Outliers** — distance to the k-th nearest neighbour. High means unusual, but unusual
is ambiguous: corrupted capture, weird camera angle, or genuinely novel behaviour you'd
be mad to drop. The vector space flags it; kinematics and annotations disambiguate.

### Measure things

**Redundancy** — threshold `S` at ~0.95, take connected components. "These 40 episodes
are effectively 8" is concrete and checkable. Easiest track-1 win available.

**Density** — count neighbours within radius r. Better than dedup because it lets you
*rebalance* rather than delete: sample inversely to density and you downweight
over-represented regions without throwing anything away.

**Volume** — Vendi score, or log-determinant of the kernel. "How much space do these
points fill." Your diversity number.

### Select things

Greedy coverage: repeatedly add whichever episode is least well covered by what's
already picked. Formally submodular, so greedy carries a `1−1/e` guarantee. ~10 lines.

### Label things

**An embedding turns any labelling problem into a 200-example problem.**

Hand-label 200 episodes for anything — success, occlusion, bad lighting — fit logistic
regression on cached vectors, predict the other 50,000. Training takes milliseconds
because the pixels were processed once. Five classifiers in an afternoon.

### Align things — the underrated one

SigLIP2 is a *joint* image–text model. The text encoder writes sentences into **the
same 1152-dimensional space**:

```python
t = model.encode_text(["a person folding a shirt",
                       "a person holding a shirt still"])
scores = X @ t.T          # every episode scored against every phrase
```

- **Mislabel detection** — score each episode against its own `task_description`. Low
  scores are episodes whose video doesn't match their label. A curation signal with
  zero annotation.
- **Zero-shot classification** — no training data. Write two phrases, compare distances.
- **Search by description** — "episodes where something gets dropped", over unlabelled data.

Track 2 rules text out of the *diversity metric*. It says nothing about text as a
*curation* signal, and this is the cheapest high-value tool on the list.

### See time — use the 32 frames

```python
T = F @ F.T          # (32, 32) — each frame against every other frame
```

Read it as a picture:

- **Bright diagonal band** → steady progress, never revisits a state. Clean episode.
- **Distinct blocks** → clear phases (approach, grasp, fold). Should line up with the
  annotation spans; if they don't, one of the two is wrong.
- **Bright patch off the diagonal** → the person returned to a state they'd already
  been in. That's a **retry**, and a retry is the visible fingerprint of failure.

Track-3 evidence you can point at on screen, from one matmul, no reward model involved.

---

## 6. What the space cannot tell you

**It doesn't know success.** SigLIP2 learned from captions, and captions describe
what's visible, not whether an attempt worked. A neat fold and a botched fold sit ~0.98
apart. Not fixable with a better prompt — it's why track 3 needs a progress signal or
hand labels.

**It has no concept of motion.** A single frame has no velocity; a mean of frames has
no time. Two episodes with the same objects and scene but completely different motion
strategies are near-identical. This is why kinematics is a *separate* view, not a
redundant one — it sees what the vision model structurally cannot.

**It over-weights scene and camera.** Background, lighting, resolution, and capture
device push vectors around, often harder than behaviour does. Mean-centre per `lab`
before scoring or you'll publish a diversity metric that mostly measures hardware.

**Similarity has no absolute meaning.** 0.8 cosine means nothing alone. Only ranks and
relative comparisons are trustworthy, and every threshold must be calibrated on your
own data — never copied from a paper.

---

## 7. You have more than one space

| Space | Dims | Sees | Blind to |
|---|---|---|---|
| Visual (SigLIP2) | 1152 | objects, scene, layout | motion, success, time |
| Kinematic (stage 1) | ~20 | smoothness, speed, hesitation | what's in the scene |
| Temporal (32×32) | — | phases, retries, stalls | absolute quality |
| Text (same as visual) | 1152 | intent, labels | anything unstated |

Compute your metric in each independently and report the vector. **The disagreements
are the finding.**

---

## 8. The tracks, concretely

### Track 1 — Curation Engine

**IN:** `X (500, 1152)` and `K (500, 20)`. No model, no GPU.

```python
S = X @ X.T                     # how similar is everything to everything
quality = f(K)                  # smooth motion, not idle, sensible duration
keep = greedy_coverage(S, quality, budget=200)
```

**OUT** — `keep_drop.csv`:

```
hash                        keep   quality  coverage  reason
2026-03-14-10-22-51-330192  KEEP     0.81     0.42    novel motion pattern
2026-03-14-10-31-02-118773  DROP     0.79     0.03    dup of 10-22-51 (cos 0.97)
2026-03-14-11-02-44-902311  DROP     0.12     0.55    jerk outlier, 4.1σ
2026-03-14-11-18-07-441290  KEEP     0.66     0.38    only left-handed demo in cluster
```

A spreadsheet with a reason per row. Plus the chart: train a small policy on your 200
vs. 200 random, show held-out error is lower.

### Track 2 — Diversity

**IN:** two lists of hashes, plus `X` and `K`. **PROCESS:** eigenvalues of the
similarity kernel, ~30 lines.

**OUT:**

```
                    episodes   Vendi(visual)   Vendi(kinematic)
Subset A                 500            47.3               12.1
Subset B                 500            44.9               38.6
```

**Read it:** A and B look equally varied *visually* — same rooms, same shirts. But A's
effective motion diversity is 12 against B's 38. **A is 500 recordings of essentially
12 different ways of moving.** B teaches a policy three times as much.

That contradiction between columns is the entire track.

### Track 3 — Human Reward Model

**IN:** `vip_progress (32,)` — how close each frame is to the goal state.

```
ep_A: [0.02, 0.09, 0.21, 0.34, 0.51, 0.67, 0.79, 0.88, 0.91, 0.94]  rises, stays
ep_B: [0.03, 0.11, 0.28, 0.44, 0.62, 0.71, 0.38, 0.15, 0.29, 0.41]  peaks, collapses
```

**PROCESS:** six features per curve (final value, peak, drawdown, monotonicity,
plateau length, slope), hand-label 200 episodes, fit logistic regression. Milliseconds.

**OUT:**

```
hash                        p_success   evidence
2026-03-14-10-22-51-330192      0.94    monotonic, ends 0.94
2026-03-14-11-02-44-902311      0.18    peaked frame 19, dropped 0.56  ← dropped the shirt
2026-03-14-11-33-51-220984      0.41    plateau frames 8-24            ← stalled
```

**Prevalence audit** — one `groupby`:

```
lab       episodes   flagged failure
aria           312       38  (12.2%)
mecka          188       51  (27.1%)   ← in-the-wild data fails twice as often
```

**Confidence meter** — the curve, plotted under the video scrubber. It already exists;
you just draw it.

---

## 9. Summary table

| | IN | model? | OUT |
|---|---|---|---|
| Kinematics | `(2847, 7)` poses | no | 20 floats |
| Featurizer | 32 JPEGs | **yes, once** | `(32, 1152)` |
| Track 1 | `(500, 1152)` | no | CSV of keep/drop + reasons |
| Track 2 | two subsets | no | 2 scores + a bar chart |
| Track 3 | `(32,)` curves | no | label + confidence per episode |

The model runs one time and produces coordinates. **Your project is the four rows
where the "model?" column says no.**
