export const CLIPS_API =
  import.meta.env.VITE_CLIPS_API ?? "https://cbstonesifer--egoverse-clips-api.modal.run";

export const clipUrl = (episodeHash: string) => `${CLIPS_API}/clip/${episodeHash}.mp4`;

export type Episode = {
  episode_hash: string;
  operator: string;
  decision: "keep" | "drop";
  score: number;
  reason: string;
  trim_start: number | null;
  trim_end: number | null;
  n_frames: number;
  act_span: number | null;
  progress_dip: number | null;
  low_detail_frac: number | null;
  right_sparc: number | null;
  right_jerk_rms: number | null;
  task: string | null;
  scene: string | null;
  num_frames: number | null;
  has_clip: boolean;
};

// Only these four are claims about the data. `below_operator_quota` is a budget decision.
export const INTEGRITY_REASONS = [
  "frame_count_mismatch",
  "no_visible_change",
  "mostly_dead_time",
  "too_short",
] as const;

export const REASON_LABELS: Record<string, string> = {
  frame_count_mismatch: "Frame count mismatch",
  no_visible_change: "No visible change",
  mostly_dead_time: "Mostly dead time",
  too_short: "Too short",
  below_operator_quota: "Below operator quota",
};

export const REASON_EXPLANATIONS: Record<string, string> = {
  frame_count_mismatch:
    "The frame count in the image shard disagrees with the catalog. One of them is lying, so the episode is untrustworthy.",
  no_visible_change:
    "Frame-to-final-frame similarity never dips (< 0.02). Nothing measurably happened in the scene.",
  mostly_dead_time: "Less than 35% of the episode contains hand motion.",
  too_short: "Fewer than 60 frames.",
  below_operator_quota:
    "No defect found — this episode scored below the keep quota for its own operator at the chosen keep rate.",
};

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
    (INTEGRITY_REASONS as readonly string[]).includes(e.reason),
  ).length;
  return {
    total: episodes.length,
    kept,
    dropped,
    integrity,
    quota: dropped - integrity,
    keepRate: episodes.length ? kept / episodes.length : 0,
  };
}

export function countByReason(episodes: Episode[]) {
  const counts = new Map<string, number>();
  for (const episode of episodes) {
    if (episode.decision !== "drop") continue;
    counts.set(episode.reason, (counts.get(episode.reason) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([reason, count]) => ({ reason, label: REASON_LABELS[reason] ?? reason, count }))
    .sort((a, b) => b.count - a.count);
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
