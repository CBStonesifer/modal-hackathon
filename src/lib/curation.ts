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

// Quality axes. A ranked drop is named by the axis it fails worst; a reason can name two.
export const QUALITY_AXES = [
  "poor_image",
  "jittery",
  "unsteady_camera",
  "hesitant",
  "incomplete",
] as const;

export const BELOW_AVERAGE = "below_average";

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
    "The frame count in the image shard disagrees with the catalog. One of them is lying, so the episode is untrustworthy.",
  no_visible_change:
    "Frame-to-final-frame similarity never dips (< 0.02). Nothing measurably happened in the scene.",
  mostly_dead_time: "Less than 35% of the episode contains hand motion.",
  too_short: "Fewer than 60 frames.",
  poor_image:
    "Faint or low-detail frames, or unstable frame size — low_detail_frac, faint_frac, size_cv.",
  jittery: "Rough hand and head motion — SPARC on all three tracks.",
  unsteady_camera: "Head moves roughly or sits idle — head_jerk_rms, head_idle_frac.",
  hesitant: "The hands spend a lot of the episode not moving — left and right idle_frac.",
  incomplete:
    "Activity does not span the episode, or the reach is indirect — act_span, straightness.",
  below_average:
    "No axis crossed its flag threshold. This episode ranked below its operator's quota without an identifiable defect — a budget decision, not a claim about the data.",
};

/** Reasons combine as "hesitant+jittery", so an axis filter matches either half. */
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
  const belowAverage = episodes.filter((e) => e.reason === BELOW_AVERAGE).length;
  const flaggedKeeps = episodes.filter(
    (e) => e.decision === "keep" && parseFlags(e.flags).length > 0,
  ).length;
  return {
    total: episodes.length,
    kept,
    dropped,
    integrity,
    belowAverage,
    quality: dropped - integrity - belowAverage,
    flaggedKeeps,
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

/** Counted by containment, so an episode named "hesitant+jittery" lands in both axes. */
export function countAxes(episodes: Episode[]) {
  const drops = episodes.filter((e) => e.decision === "drop");
  return QUALITY_AXES.map((axis) => ({
    reason: axis as string,
    label: REASON_LABELS[axis],
    count: drops.filter((e) => reasonMatches(e.reason, axis)).length,
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
