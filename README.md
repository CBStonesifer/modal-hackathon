# modal-hackathon

Automated curation for egocentric robot demonstration data, and a dashboard for auditing what it
decided.

The corpus is [EgoVerse](https://github.com/GaTech-RL2/EgoVerse) — head-mounted video plus hand and
head poses from human demonstrators, stored as Zarr in Cloudflare R2 with a Postgres catalog. The
working slice is `mecka / cup_on_saucer`: **5,407 episodes**. The question the code answers is
*which of these episodes should a policy actually be trained on*, and the constraint is that the
answer has to be defensible — every rejection names a measurable reason, and no rejection may
reduce to "this demonstrator is unusual".

```
video-curation/   Modal jobs + local scoring — R2 to a keep/drop manifest
dashboard/        TanStack Start UI — audit the manifest and watch the clips
```

---

## The pipeline

Four tiers, each an order of magnitude more expensive than the last, each narrowing what the next
has to look at.

| tier | code | reads | cost per episode | output |
|---|---|---|---|---|
| 0 | `explore.py index` | Postgres catalog | — | `episodes.parquet` |
| 1 | `curate.py::main` | Zarr shard index only | 1 LIST + one 16 KB ranged GET | `frames.parquet` |
| 2 | `curate.py::poses` | pose arrays (no images) | ~1 MB download | `kinematics.parquet` |
| 3 | `curate.py::embed` | 32 sampled frames, L4 GPU | seconds of GPU | `emb_*.npz` |
| — | `score.py` | all of the above | local, seconds | `keep_drop.csv` |

**Tier 1 never downloads a pixel.** The Zarr v3 shard index is a table of `(offset, nbytes)` pairs
at the tail of each shard, so one ranged GET of ~16 KB yields the exact compressed size of every
frame in the episode. Compressed JPEG size tracks image detail (r=+0.95 against Laplacian
variance), which is enough to find faint frames, unstable encoding, and a frame count that
disagrees with what SQL claims. Whole-slice scan cost: cents.

**Tier 2** pulls the pose arrays and derives motion features per track (left hand, right hand,
head): SPARC smoothness, jerk RMS, idle fraction, path straightness, and the motion-energy span —
the sub-interval that actually contains movement, thresholded against the episode's own peak speed
rather than a percentile.

**Tier 3** runs DINOv2 or SigLIP2 on an L4, sampling 32 frames per episode weighted toward the
active span, optionally cropped to the projected hand positions.

## How the decision is made — `score.py`

Two independent mechanisms, and they are deliberately kept separate.

**Integrity failures** are verifiable claims about the data. Four rules, each independently
checkable: fewer than 60 frames; a shard frame count that disagrees with the catalog; no visible
change across the episode; less than 35% of the episode containing hand motion.

**Statistical outliers** are found in the embedding space, not by a quota. Episodes are clustered
(k chosen by silhouette), and each is tested for Mahalanobis distance to its own cluster centre
against chi-square, with Benjamini–Hochberg controlling the false-drop rate across all simultaneous
tests. Two details matter:

- The covariance estimator is **MCD**, not `np.cov`, because a plain covariance is estimated *from*
  the outliers and hides them.
- The chi-square null is **rescaled by simulation**. MCD distances are only asymptotically
  chi-square; at these sample sizes the uncalibrated version flagged 18% of everything. The same
  estimator is fit to Gaussian data of identical shape to recover the scale factor, matched at the
  99th percentile because that is where the decisions are made.

So the number kept is whatever the data supports, not a dial.

### Guarding against the operator confound

The dominant axis of variation in this corpus is *who recorded it*. A curator that scores on
demonstrator identity is a demonstrator detector wearing a lab coat, so two defences are wired in:

1. **Feature selection is measured, not chosen.** Any feature whose variance is more than 35%
   explained by `operator` (η²) is excluded from the ordering score outright.
2. **Everything remaining is z-scored within operator** — median-centred, clipped to ±4.

The episode descriptor used for clustering is built from *within-episode* comparisons only
(similarity to the episode's own final frame, plus step magnitudes), which cancels the appearance
offset carried by scene and demonstrator. That offset is 98% of the raw embedding's
nearest-neighbour structure. The raw mean-embedding descriptor is kept behind `--repr raw` so the
confound can be measured rather than asserted.

### Naming the failure

A drop that says only "ranked low" is not auditable, so each one is labelled by which of five
interpretable axes it fails worst. Measured pairwise correlation between the axes is ≤ 0.14, and
k-means over the drop population recovers roughly one cluster per axis — they are distinct modes,
not five names for one thing.

| axis | built from | flagged when |
|---|---|---|
| `poor_image` | `low_detail_frac`, `faint_frac`, `size_cv` | z > 1.0 |
| `jittery` | SPARC on left hand, right hand, head | z > 1.0 |
| `unsteady_camera` | `head_jerk_rms`, `head_idle_frac` | z > 1.0 |
| `hesitant` | left and right `idle_frac` | z > 1.0 |
| `incomplete` | `act_span`, `right_straightness` | z > 1.0 |

Each axis is the mean of its features' within-operator z-scores, signed so positive always means
worse. Zero is typical for that demonstrator. An episode that crosses no threshold but still fails
the outlier test is labelled `below_average` rather than given a defect it does not have — and a
*kept* episode that crosses a threshold keeps its warning in the `flags` column instead of having
it discarded.

## Does any of it help? — `ablate.py`

Curation that improves no downstream metric is decoration. `ablate.py` fits ridge regression to
predict the next end-effector displacement from a short pose history — the poses *are* the actions
in this dataset, so it is a stripped-down version of what a behaviour-cloning policy learns — and
sweeps keep-fraction against random subsets of matched size. Closed form, so a 12-point sweep with
5 seeds is seconds.

The evaluation split is **by operator**, fixed across every condition. A random split leaks:
near-duplicate retakes from one demonstrator would land on both sides and every condition would
look fine.

## Serving the footage — `clips.py`

R2 → Modal Volume → one public endpoint, so the browser needs no credentials.

```
modal run clips.py::sync              # copy preview mp4s into the volume (resumable)
modal run clips.py::build_manifest    # keep_drop.csv + catalog -> manifest.json
modal deploy clips.py                 # serve both
```

| route | returns |
|---|---|
| `GET /manifest.json` | every episode with its decision, reason, score and axis z-scores |
| `GET /clip/{episode_hash}.mp4` | the preview clip; `Range` honoured, so seeking works |

`episode_hash` is matched against `^[0-9a-f]{24}$` before anything touches the filesystem — it is a
public endpoint taking a path segment. CORS is open because the payload is read-only public data.

---

## Dashboard

`dashboard/` — TanStack Start, React 19, Tailwind v4, shadcn/ui, recharts. It fetches
`manifest.json` once and does all filtering client-side.

- **Overview** — keep/drop counts split into integrity failures, named quality failures, and
  below-average; drop reasons charted; keep rate per operator; score distribution with keep and
  drop overlaid.
- **Clips** — every episode, filterable by status, by specific failure reason, and by operator.
  Opening one plays the clip beside the metrics that drove the decision and a bar per failure axis,
  centred on zero and highlighted past the flag threshold.

```sh
cd dashboard && bun install && bun run dev
```

The API base defaults to the deployed Modal URL and can be overridden with `VITE_CLIPS_API`.

---

## Reproducing

```sh
cd video-curation
./explore.py setup                          # bootstrap credentials -> ~/.egoverse_env
./explore.py index                          # catalog -> episodes.parquet
modal run --detach curate.py --limit 0      # tier 1
modal run --detach curate.py::poses --limit 0
modal run --detach curate.py::embed --sample 0 --encoder dinov2
./score.py --frames frames_episodes.parquet --kinematics kinematics.parquet
./ablate.py
```

Credentials live only in `~/.egoverse_env` (chmod 600) and the Modal secret `egoverse-r2`. Nothing
in this repository contains one.

`*.parquet` and `*.npz` are gitignored — together they are ~220 MB and every one is regenerable
from the commands above. `keep_drop.csv` and `ablation.csv` are committed, since they are the
outputs the dashboard and the writeup refer to.

## Known state

Current manifest: **5,337 keep / 70 drop**, every drop an integrity failure, plus **2,297
episodes carrying a warning flag** without being rejected. The dashboard and the endpoint both
reflect this.

**Tier 3 has only run on 350 of 5,407 episodes.** The outlier test therefore examined 318 and
found nothing at α=0.01, and `progress_dip` is null for the remaining 94%. This is the one number
that matters for reading the dashboard honestly: 5,089 episodes are marked keep because nothing
rejected them, not because anything cleared them. The Overview states this in a coverage banner,
the Clips tab has a **Never outlier-tested** filter, and per-episode `cluster` / `outlier_p` read
"not tested" rather than showing a blank. Running `curate.py::embed` across the whole slice is
what closes it.

Two smaller things `score.py` still does that are worth knowing: the per-axis reference
distributions are computed over the whole population, including episodes the integrity rules
already rejected, and trim spans come from the right hand alone, so a left-lead demonstration can
get its opening clipped.

Longer design notes (`PIPELINE.md`, `RESULTS.md`, `CONCEPTS.md`, `HANDOFF.md` and others) are kept
outside this repository. They predate the quota removal and still describe per-operator quotas
throughout, so this README is the current account of how the engine works.
