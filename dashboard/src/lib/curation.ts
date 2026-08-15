export const CLIPS_API =
  import.meta.env.VITE_CLIPS_API ?? "https://cbstonesifer--egoverse-clips-api.modal.run";

export const clipUrl = (episodeHash: string) => `${CLIPS_API}/clip/${episodeHash}.mp4`;

export type Episode = {
  episode_hash: string;
  operator: string;
  decision: "keep" | "drop";
  score: number;
  reason: string | null;
  flags: string | null;
  trim_start: number | null;
  trim_end: number | null;
  n_frames: number;
  act_span: number | null;
  progress_dip: number | null;
  cluster: number | null;
  outlier_p: number | null;
  ax_poor_image: number | null;
  ax_jittery: number | null;
  ax_unsteady_camera: number | null;
  ax_hesitant: number | null;
  ax_incomplete: number | null;
  task: string | null;
  scene: string | null;
  num_frames: number | null;
  has_clip: boolean;
};

// Data-integrity failures. Each is independently verifiable against the episode.
export const INTEGRITY_REASONS = [
  "frame_count_mismatch",
  "no_visible_change",
  "mostly_dead_time",
  "too_short",
] as const;

// Quality axes. An episode past the flag threshold on one carries a warning; an episode the
// outlier test rejected is named by whichever of these it fails worst.
export const QUALITY_AXES = [
  "poor_image",
  "jittery",
  "unsteady_camera",
  "hesitant",
  "incomplete",
] as const;

export const BELOW_AVERAGE = "below_average";
export const AXIS_FLAG = 1.0; // z above this counts as failing the axis, matching score.py

export const AXIS_FIELD: Record<string, keyof Episode> = {
  poor_image: "ax_poor_image",
  jittery: "ax_jittery",
  unsteady_camera: "ax_unsteady_camera",
  hesitant: "ax_hesitant",
  incomplete: "ax_incomplete",
};

export const REASON_LABELS: Record<string, string> = {
  frame_count_mismatch: "Frame count mismatch",
  no_visible_change: "No visible change",
  mostly_dead_time: "Mostly dead time",
  too_short: "Too short",
  poor_image: "Poor image",
  jittery: "Jittery",
  unsteady_camera: "Unsteady camera",
  hesitant: "Hesitant",
  incomplete: "Incomplete",
  below_average: "Below average",
};

export const REASON_EXPLANATIONS: Record<string, string> = {
  frame_count_mismatch:
    "The number of images in the storage shard disagrees with the number the database claims. One of them is wrong, so the episode cannot be trusted.",
  no_visible_change:
    "Every frame was compared against the episode's final frame. The similarity never dropped by more than 0.02, meaning nothing measurably happened.",
  mostly_dead_time:
    "Less than 35% of the episode falls between the first and last real hand movement.",
  too_short: "Fewer than 60 frames.",
  poor_image:
    "Faint or low-detail frames, or an unstable frame size — measured from compressed JPEG byte size.",
  jittery: "Rough, unsmooth motion — SPARC smoothness on both hands and the head.",
  unsteady_camera: "The head moves roughly, or sits still for long stretches.",
  hesitant: "The hands spend an unusual share of the episode not moving.",
  incomplete:
    "Movement does not span the episode, or the reach wanders instead of going straight to the target.",
  below_average:
    "The outlier test rejected this episode, but it crossed no axis threshold — statistically unusual without an identifiable defect.",
};

/** Plain-language note on where each dialog metric comes from. */
export const METRIC_NOTES: Record<string, string> = {
  score:
    "Average of every quality feature that survived the confound check, re-centred on this operator's own median. 0 is typical for them.",
  operator: "Which demonstrator recorded the episode. All scoring is relative to this person.",
  frames: "Images actually present in the storage shard, counted from its index.",
  catalog_frames: "Frames the database claims exist. A disagreement is a defect.",
  act_span:
    "Share of the episode between the first and last real hand movement, where 'real' means faster than 15% of this episode's own peak speed.",
  progress_dip:
    "How much the scene visibly changed, from comparing every frame's embedding to the final frame. Higher means more happened.",
  trim_span: "Frame range worth keeping — first moving frame to last.",
  cluster:
    "Which behaviour group the episode's embedding landed in. Groups are found by k-means, not defined by hand.",
  outlier_p:
    "Probability of sitting this far from its own group's centre by chance. Small means suspicious.",
};

/** Plain-language note on how each failure axis is measured. */
export const AXIS_NOTES: Record<string, string> = {
  poor_image: "faintness and detail loss, from compressed JPEG size per frame",
  jittery: "SPARC motion smoothness across left hand, right hand and head",
  unsteady_camera: "head jerk, plus how long the head sat completely still",
  hesitant: "share of the episode with both hands stationary",
  incomplete: "whether movement spans the episode, and how directly the hand reaches",
};

export const AXIS_METHOD =
  "Each axis averages its features after re-centring on this operator's median and dividing by their spread. Positive is worse than that operator's typical; past 1.0 it is flagged.";

/** Reasons can combine as "hesitant+jittery", so an axis matches either half. */
export function reasonMatches(reason: string | null, selection: string): boolean {
  if (!reason) return false;
  if ((QUALITY_AXES as readonly string[]).includes(selection)) {
    return reason.split("+").includes(selection);
  }
  return reason === selection;
}

export function parseFlags(flags: string | null): string[] {
  return flags ? flags.split("+").filter(Boolean) : [];
}

export function hasFlag(episode: Episode, axis: string): boolean {
  return parseFlags(episode.flags).includes(axis);
}

export async function fetchEpisodes(): Promise<Episode[]> {
  const response = await fetch(`${CLIPS_API}/manifest.json`);
  if (!response.ok) {
    throw new Error(`manifest unavailable (${response.status})`);
  }
  return response.json();
}

export function summarise(episodes: Episode[]) {
  const kept = episodes.filter((e) => e.decision === "keep").length;
  const dropped = episodes.length - kept;
  const integrity = episodes.filter((e) =>
    (INTEGRITY_REASONS as readonly string[]).includes(e.reason ?? ""),
  ).length;
  const flagged = episodes.filter((e) => parseFlags(e.flags).length > 0).length;
  // an episode only reaches the outlier test if it has an embedding; the rest were never examined
  const tested = episodes.filter((e) => e.cluster !== null).length;
  return {
    total: episodes.length,
    kept,
    dropped,
    integrity,
    outlier: dropped - integrity,
    flagged,
    flaggedKeeps: episodes.filter((e) => e.decision === "keep" && parseFlags(e.flags).length > 0)
      .length,
    tested,
    untested: episodes.length - tested,
    keepRate: episodes.length ? kept / episodes.length : 0,
  };
}

export function countIntegrity(episodes: Episode[]) {
  return INTEGRITY_REASONS.map((reason) => ({
    reason: reason as string,
    label: REASON_LABELS[reason],
    count: episodes.filter((e) => e.reason === reason).length,
  })).sort((a, b) => b.count - a.count);
}

/**
 * Warning flags across every episode, kept or dropped. Counted by containment, so an episode
 * flagged "hesitant+jittery" lands in both axes.
 */
export function countFlags(episodes: Episode[]) {
  return QUALITY_AXES.map((axis) => ({
    reason: axis as string,
    label: REASON_LABELS[axis],
    count: episodes.filter((e) => hasFlag(e, axis)).length,
  })).sort((a, b) => b.count - a.count);
}

export function keepRateByOperator(episodes: Episode[]) {
  const tally = new Map<string, { kept: number; total: number }>();
  for (const episode of episodes) {
    const entry = tally.get(episode.operator) ?? { kept: 0, total: 0 };
    entry.total += 1;
    if (episode.decision === "keep") entry.kept += 1;
    tally.set(episode.operator, entry);
  }
  return [...tally.entries()]
    .map(([operator, { kept, total }]) => ({
      operator: operator.slice(0, 8),
      total,
      keepRate: (kept / total) * 100,
    }))
    .sort((a, b) => a.keepRate - b.keepRate);
}

export function scoreHistogram(episodes: Episode[], bins = 24) {
  if (!episodes.length) return [];
  const scores = episodes.map((e) => e.score);
  const low = Math.min(...scores);
  const high = Math.max(...scores);
  const width = (high - low) / bins || 1;

  const buckets = Array.from({ length: bins }, (_, index) => ({
    score: Number((low + width * (index + 0.5)).toFixed(2)),
    keep: 0,
    drop: 0,
  }));

  for (const episode of episodes) {
    const index = Math.min(bins - 1, Math.floor((episode.score - low) / width));
    buckets[index][episode.decision] += 1;
  }
  return buckets;
}

export function operators(episodes: Episode[]) {
  return [...new Set(episodes.map((e) => e.operator))].sort();
}
