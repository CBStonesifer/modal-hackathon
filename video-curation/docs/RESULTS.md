# A Curation Engine for EgoVerse — and why most of the obvious signals are wrong

**Slice:** `lab=mecka`, `task=cup_on_saucer` — 5,409 episodes, 22 demonstrators, 904,511 frames,
~63 GB of video.
**Deliverable:** `keep_drop.csv` — a keep/drop decision, a score, a machine-readable reason, and
a trim span for every episode.
**Cost:** 5.9 MB of derived features. Under 20 minutes of compute. ~$4.

---

## 1. The headline

Every obvious quality signal in this dataset is substantially a measurement of **who recorded
the episode**, not how well it was performed. We measured that rather than assuming it, and used
the measurement to decide which features are allowed into the score.

| candidate signal | variance explained by demonstrator |
|---|---|
| mean JPEG size per frame | **84.3%** |
| hand span | **84.3%** |
| episode duration | **70.5%** |
| **jerk RMS** (the standard smoothness metric) | **42.6%** |
| nearest neighbour in DINOv2 space | **98.0%** of neighbours share the demonstrator |
| SPARC (spectral arc length) | 23.8% |
| activity span | 14.7% |
| head-motion smoothness | 13.3% |

A curation engine built on any of the top five would rank demonstrators, not demonstrations —
and would look like it was working. That is the failure mode this system is designed around.

---

## 2. What was built

Four tiers, each filtering what the next must touch. Tiers 1–2 use no GPU and no model.

| tier | what it reads | bytes moved | time (5,407 eps) |
|---|---|---|---|
| 0 · catalog | Postgres `app.episodes` | 40 MB | seconds |
| 1 · per-frame quality | **last 16 KB of each image shard** | 87 MB | **66 s** |
| 2 · kinematics | pose arrays only (no pixels) | 1.5 GB | 6m23s |
| 3 · semantics | 32 hand-cropped frames per episode, DINOv2 on an L4 | ~2 MB/ep | 6m45s / 400 eps |

### The trick that makes tier 1 nearly free

Each episode's frames are stored in one sharded Zarr object (~12 MB) with a
`1000 × (offset, nbytes)` index in its **final 16,004 bytes**. A single ranged GET of that tail
yields the exact compressed size of every frame without transferring one pixel — verified
`corr = 1.00000` against the real JPEG lengths.

**904,511 per-frame measurements for 87 MB instead of 63 GB**, and the same index gives byte
offsets for random-access frame fetching in tier 3, making it 5× cheaper than downloading shards.

---

## 3. Findings that shaped the design

### 3.1 Visual and motion quality are independent

`corr(low_detail_frac, head_jerk_rms) = +0.024`; every visual↔kinematic pairing is |r| ≤ 0.09.
They are two genuinely separate views, so the score uses both. Filtering on one tells you
nothing about the other.

### 3.2 There is no single quality axis

PCA over 50 features: PC1 explains **13%**, six components reach 52%. Quality here is
irreducibly multi-dimensional; a single scalar necessarily discards most of the variance. The
score is reported alongside its components for that reason.

### 3.3 Use SPARC, not jerk

Both measure smoothness. Jerk RMS is **42.6%** demonstrator; SPARC is **23.8%**. SPARC is
normalised for movement amplitude and duration, so it survives the confound check and jerk does
not. Jerk is the more commonly cited metric in the demonstration-curation literature, and on
multi-demonstrator data it is the worse choice.

### 3.4 Cross-episode visual comparison does not work on this slice — and cannot be repaired

Four independent attempts to break the demonstrator confound in DINOv2 space:

| attempt | NN shares demonstrator |
|---|---|
| 16 frames, uniform | 97.3% |
| 32 frames, motion-weighted | 98.0% |
| frame-delta (motion-only representation) | 85.4% |
| **hand-cropped to projected MANO keypoints** | **97.8%** |
| *(chance)* | *8.9%* |

Hand-cropping fails for a reason we had already measured: **hand span is 84.3% demonstrator**.
The hands *are* the person. Cropping to them crops to the most identifying object in the frame —
median pairwise similarity actually *rose* (0.485 → 0.538).

The cause is the data, not the method. Repeating the identical pipeline on a **58-scene** slice:

| | task effect | scene effect | ratio |
|---|---|---|---|
| raw DINOv2 | +0.324 | +0.136 | **2.38** |
| frame-delta | +0.036 | −0.000 | **91.3** |

*(task effect = similarity of same-task-different-room pairs minus different-task-different-room pairs)*

With scene variety present, DINOv2 recognises the activity at 2.4× the room. `cup_on_saucer` is
one task, one object, one kitchen — there is simply no task variance for a vision model to
encode. **Conclusion: embedding-geometry diversity selection is unsound on single-scene slices,
and no representation trick recovers a signal the data never contained.**

### 3.5 The rule this yields

| comparison | demonstrator confound | usable |
|---|---|---|
| **between** episodes — diversity, coverage selection | fatal | ✗ |
| **within** an episode — progress, trimming, retries | cancels exactly | ✓ |

Within one episode the demonstrator, rig, room, and lighting are constant, so they cancel. This
is why the system's semantic features are all within-episode.

### 3.6 The within-episode progress signal is real

Mean cosine similarity of each frame to its own final frame, 350 episodes, hand-cropped:

```
0.679  0.675  0.663  0.644  0.610  0.667  0.756  0.838
       ←—— diverging ——→    ^dip    ←—— converging ——→
```

A clean U. The scene starts resembling its own end state (cup and saucer laid out, static),
diverges as hands enter and occlude, converges on the goal. **The dip is the manipulation.**

Hand-cropping deepens the dip by 32% (−0.132 → −0.174), confirming the crop is working even
though it does nothing for cross-episode comparison. The measurement reproduces across two
sampling schemes and two independent samples (7.0% vs 7.7% "nothing happened" rate), so it is
not a sampling artefact.

### 3.7 Trimming matters for a tail, and two signals catch opposite ends

Median dead time is 2.5% (0.14 s of 5.7 s) — these clips are already tight. But 14.9% carry
>20% dead time and 5.0% carry >40%.

Splitting all 904,511 frames by the pose-derived activity boundary:

| zone | frames | mean detail z | faint rate |
|---|---|---|---|
| dead head | 54,248 | **+0.166** | 8.7% |
| active | 825,560 | −0.027 | 14.5% |
| dead tail | 24,703 | **−0.307** | 23.9% |

The dead tail has 1.65× the faint rate of the active zone — an independent confirmation of the
pose boundary. But the dead head has *more* detail than the active zone: at the start the scene
is fully set up and static; at the end the hands have withdrawn. **Motion energy is the only
signal that finds the dead head; image detail independently confirms the dead tail.** Neither
alone trims both, which is why they are only weakly correlated at the episode level (+0.158) and
why both are computed.

---

## 4. The engine

```
1. HARD DROPS — data integrity, each independently verifiable
     no image shard · n_frames < 60 · frame count disagrees with SQL
     · no visible change (progress dip < 0.02) · >65% dead time

2. FEATURE SELECTION — automatic, by measured confounding
     exclude any feature with eta^2 vs demonstrator > 35%
     -> excluded: right_jerk_rms (43%), left_jerk_rms (45%), left_straightness (36%)
     -> retained: 14 features spanning visual, kinematic and semantic

3. SCORE — z-normalised WITHIN demonstrator, then averaged
     so a fast worker is compared against their own distribution, not a global one

4. QUOTA — keep the top N% WITHIN each demonstrator
     so no one can be disproportionately deleted

5. TRIM — one index range per episode, applied to poses and frames identically
```

Step 5 matters because the poses *are* the actions: a manifest that trims frames while leaving
pose arrays whole silently destroys the observation↔action correspondence a policy trains on.

### Output

```csv
episode_hash,decision,score,reason,flags,trim_start,trim_end,ax_poor_image,ax_jittery,ax_unsteady_camera,ax_hesitant,ax_incomplete
692e98937641,drop,0.3884,too_short,,,,-0.19,-1.49,0.93,-0.29,-2.32
692eeefcef75,drop,0.2265,mostly_dead_time,,,,0.41,0.12,-0.33,0.88,2.71
692ea23dc621,keep,1.2041,,hesitant,3,168,-0.42,-0.61,0.08,1.24,-0.55
```

At a 70% keep target: **3,732 keep / 1,675 drop**.

### Every drop is named, and five failure modes are independent

Earlier versions labelled 96% of drops `below_operator_quota`, which is not a reason. Five axes,
z-scored within demonstrator, with **maximum pairwise correlation 0.14** — and k-means over the
drop population recovers exactly one cluster per axis, so they are distinct modes rather than
five names for one thing:

| reason | n | what it means |
|---|---|---|
| `hesitant` | 211 | long idle stretches |
| `poor_image` | 196 | low detail, faint frames, unstable exposure |
| `jittery` | 174 | poor SPARC on hands and head |
| `unsteady_camera` | 151 | head jerk and head idle |
| `incomplete` | 112 | short active span, low straightness |
| two-axis combinations | 437 | e.g. `hesitant+jittery` |
| `below_average` | 248 | **no identifiable defect** |
| hard drops | 63 | 34 frame-count mismatch, 22 no-visible-change, 4 mostly-dead-time, 3 too-short |

Three things this exposes:

- **15.4% of ranked drops have no defect at all** — they are below their demonstrator's median
  and nothing more. That share of the drop budget buys nothing, and saying so is more useful
  than hiding it behind a quota label.
- **24.4% of *kept* episodes carry a warning flag**, surfaced in a `flags` column so a
  downstream user can tighten without re-running the pipeline.
- **Multi-axis failure is a confidence signal.** 437 dropped episodes fail two axes and 89 fail
  three or more, against 50 and 0 respectively among kept episodes.

All five axis z-scores ship in the manifest, so every decision is inspectable.

### Fairness check

Keep-rate by demonstrator spans **63.6% – 70.0%** across all 22, against a 70% target. The
spread is entirely accounted for by hard drops. A duration threshold — the obvious naive rule —
would instead have removed a disproportionate share of the two demonstrators who work fastest,
who between them account for 26% of the slice.

---

## 4b. We tried to find the threshold empirically. There isn't one.

`ablate.py` builds the proxy-policy ablation the brief asks for: ridge regression predicting
future end-effector displacement, evaluation set fixed and split **by demonstrator**, 5 seeds per
point. Three results, in order:

- **The first proxy was saturated.** Next-frame prediction gave identical held-out MSE at 313 and
  3,131 episodes. Not "curation doesn't help" — a 15-parameter model learns "hands move smoothly"
  from 300 episodes and cannot use more.
- **Fixing it barely helped.** Longer horizons (0.5 s, 1.0 s) plus 1024 random Fourier features
  finally made data quantity matter — by **1.3%–2.6%** between 10% and 100% of the data. Quality
  ranking captured ~1% of that, at 1.2σ. Not significant at any horizon or keep fraction.
- **The score distribution has no natural break.** Unimodal, one histogram maximum, knee at 92%.
  And calibration against the 63 independently-verifiable bad episodes is weak: their median
  percentile is 37.3% against 50.1% for everything else, and cutting the bottom 10% catches only
  27% of them.

**So no threshold is intrinsic to this data.** The defensible cuts are the 63 hard drops; the 257
episodes (4.8%) failing *both* independent views, which is 28% above the 201 expected by chance;
and beyond that, any percentile — explicitly a budget decision.

## 4c. A stronger encoder does not fix the confound

Same 400 episodes, 32 hand-cropped motion-weighted frames:

| encoder | params | NN-same-demonstrator | progress dip | never-dip |
|---|---|---|---|---|
| dinov2-base | 86M | 97.8% | −0.174 | 4.9% |
| dinov2-large | 300M | **98.0%** | −0.186 | 6.3% |
| **siglip2** | 375M | 95.2% | **−0.195** | **2.3%** |

*(chance 8.5%)*

A 4.4× parameter range moves the confound from 98.0% to 95.2% — still ~11× chance. **No encoder
escapes it**, now tested across three rather than assumed from one. But SigLIP2 is clearly best
at the signal that *does* work: 12% deeper progress dip and less than half the never-dip rate.
The pipeline defaults to SigLIP2 for that reason. Capacity is not the bottleneck; the axis for a
real step change is *modality* — a video-native encoder — not size.

## 5. Limitations — stated plainly

1. **No ground truth exists.** `is_eval` is `False` on all 420,415 rows, `eval_success` is its
   dataclass default `True`, `eval_score` is `-1` everywhere. These are unpopulated schema
   placeholders. So there is **no precision/recall number here** — the thresholds are principled
   and auditable, but they have not been checked against human judgement. That is the single
   biggest gap and the obvious next step (~150 labels via the preview MP4s).
2. **The keep rate is a budget decision, and we have shown it cannot currently be derived**
   (§4b). Only the 63 hard drops and arguably the 257 two-axis failures are claims about the
   data. The word "optimal" is not defensible on this evidence; "auditable" is.
3. **One lab, one task, one encoder.** All conclusions are mecka + DINOv2. Aria data has moving
   scenes and eye gaze and may behave differently.
4. **Scene and task are entangled** in the multi-scene control — a `repair_bench` contains repair
   tasks — so the 2.38 task/scene ratio partly measures object co-occurrence.
5. **mecka ships no frame timestamps**, so its declared 30 fps is taken on faith. The Aria
   episode we checked declared 30 and measured 15.4 — a 2× error. Cross-rig smoothness
   comparison inherits that risk.
6. **`zarr.json` is unreliable metadata** — the mecka store calls its own task `"debug"`. All
   metadata is read from Postgres.

---

## 6. Why this is responsive to the brief

EgoVerse's own paper argues that *"scale alone is not sufficient — effective scaling requires
alignment between the human data and the robot's learning objective."*

The result here is that the most natural proxies for that alignment — visual similarity, motion
smoothness, episode length, embedding diversity — are, on this data, substantially measurements
of capture identity. None of them fails loudly. Each produces a plausible number and defensible
clusters.

The contribution is therefore not only the manifest but the **confound-controlled procedure that
produced it**: feature selection driven by measured η², normalisation and quotas within
demonstrator, and a hard line between within-episode signals (trustworthy) and between-episode
signals (not, on this slice).

---

## 7. Reproducing

```bash
./explore.py setup                                  # credentials, pure Python, no AWS CLI
./explore.py index                                  # -> episodes.parquet
modal run curate.py::main   --limit 0               # tier 1, 66 s
modal run curate.py::poses  --limit 0               # tier 2, 6m23s
modal run curate.py::embed  --sample 0 --hand-crop  # tier 3, GPU
./score.py --keep-frac 0.70                         # -> keep_drop.csv
```

Supporting detail in `DATA.md` (what the dataset actually contains, measured) and `PIPELINE.md`
(every experiment, including the ones that failed).
