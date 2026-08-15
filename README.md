# modal-hackathon

Automated curation for egocentric robot demonstration data, and a dashboard for auditing what it
decided.

The corpus is [EgoVerse](https://github.com/GaTech-RL2/EgoVerse) — head-mounted video plus hand and
head poses, stored as Zarr in Cloudflare R2 with a Postgres catalog. The working slice is
`mecka / cup_on_saucer`: **5,407 episodes**. The scan runs on Modal; scoring runs locally.

Current manifest: **5,337 keep · 70 drop · 2,297 carrying a warning flag**.

```
video-curation/   Modal jobs + local scoring — R2 to a keep/drop manifest
dashboard/        TanStack Start UI — audit the manifest and watch the clips
```

---

## How curation works

**Measure, in four passes — each one more expensive, each looking at less.**

- `explore.py index` — pull the Postgres catalog.
- `curate.py::main` — one 16 KB ranged read per episode gets the compressed size of every frame
  from the Zarr shard index. No pixels downloaded. Costs cents for the whole slice.
- `curate.py::poses` — download the pose arrays and derive motion features: smoothness (SPARC),
  jerk, idle fraction, reach straightness, and the span of the episode containing real movement.
- `curate.py::embed` — 32 frames per episode through a vision model on an L4, weighted toward the
  active span.

**Reject on integrity — four rules, each independently checkable.**

- `too_short` — fewer than 60 frames.
- `frame_count_mismatch` — the shard and the database disagree on how many frames exist.
- `no_visible_change` — no frame differs from the final frame by more than 0.02.
- `mostly_dead_time` — under 35% of the episode falls between the first and last hand movement.

**Reject on outlierness — statistically, not by quota.**

- Episodes are clustered on their embeddings (k picked by silhouette).
- Each is tested for distance from its own cluster centre against chi-square, with
  Benjamini–Hochberg holding the false-drop rate across all tests.
- Covariance is MCD, not plain — a plain one is estimated *from* the outliers and hides them.
- The chi-square null is rescaled by simulation, because MCD distances run large at these sample
  sizes and the uncalibrated version flagged 18% of everything.
- So the number kept is whatever the data supports, not a dial.

**Flag on quality — five axes, warnings rather than rejections.**

| axis | measured from |
|---|---|
| `poor_image` | faintness and detail loss, from compressed JPEG size |
| `jittery` | SPARC smoothness across left hand, right hand, head |
| `unsteady_camera` | head jerk, and how long the head sat still |
| `hesitant` | share of the episode with both hands stationary |
| `incomplete` | whether movement spans the episode, and how directly the hand reaches |

- Each axis averages its features after re-centring on that operator's median. Positive is worse
  than typical for that person; past **1.0** it is flagged.
- A flagged episode is still kept — the warning rides along in the `flags` column instead of being
  thrown away.

**Don't build a demonstrator detector.**

The dominant axis of variation in this corpus is *who recorded it*, so:

- Any feature whose variance is more than 35% explained by `operator` (η²) is excluded from scoring
  outright.
- Everything left is z-scored within operator.
- The clustering descriptor uses only within-episode comparisons, which cancels the appearance
  offset carried by scene and demonstrator — that offset is 98% of the raw embedding's
  nearest-neighbour structure.

**Check it actually helps** — `ablate.py` fits ridge regression to predict the next end-effector
displacement from a short pose history, and sweeps keep-fraction against random subsets of matched
size. The eval split is by operator, so near-duplicate retakes can't land on both sides.

---

## Reproduce

```sh
cd video-curation

./explore.py setup                              # credentials -> ~/.egoverse_env
./explore.py index                              # catalog -> episodes.parquet

modal run --detach curate.py --limit 0          # tier 1: frame sizes
modal run --detach curate.py::poses --limit 0   # tier 2: kinematics
modal run --detach curate.py::embed --sample 0  # tier 3: embeddings on L4

./score.py --frames frames_episodes.parquet --kinematics kinematics.parquet
./ablate.py                                     # optional: does curation help?
```

Then publish the footage and manifest:

```sh
modal run clips.py::sync              # copy preview mp4s into a Modal Volume (resumable)
modal run clips.py::build_manifest    # keep_drop.csv + catalog -> manifest.json
modal deploy clips.py
```

| route | returns |
|---|---|
| `GET /manifest.json` | every episode with its decision, reason, score and axis scores |
| `GET /clip/{episode_hash}.mp4` | the preview clip; `Range` honoured, so seeking works |

`episode_hash` is matched against `^[0-9a-f]{24}$` before anything touches the filesystem.

### Dashboard

```sh
cd dashboard && bun install && bun run dev
```

Overview shows the drop reasons, flag counts, keep rate per operator and score distribution. Clips
lists every episode — filterable by status, reason, flag and operator — and plays the footage beside
the metrics that drove the decision. API base defaults to the deployed Modal URL; override with
`VITE_CLIPS_API`.

---

## Notes

- Credentials live only in `~/.egoverse_env` (chmod 600) and the Modal secret `egoverse-r2`.
  Nothing in this repository contains one.
- `*.parquet` and `*.npz` are gitignored — ~175 MB, all regenerable above. `keep_drop.csv` and
  `ablation.csv` are committed, since they are what the dashboard and the writeup read.
- Two known rough edges in `score.py`: axis baselines are computed over the whole population
  including already-rejected episodes, and trim spans come from the right-hand side alone.
- Long-form design notes are kept outside this repo. They predate the removal of per-operator
  quotas and describe a model that no longer exists — this README is the current account.
