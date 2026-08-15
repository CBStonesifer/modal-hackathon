export const CLIPS_API =
  import.meta.env.VITE_CLIPS_API ?? "https://cbstonesifer--egoverse-clips-api.modal.run";

export const clipUrl = (episodeHash: string) => `${CLIPS_API}/clip/${episodeHash}.mp4`;

export type Episode = {
  episode_hash: string;
  operator: string;
  task: string | null;
  scene: string | null;
  decision: "keep" | "drop";
  reason: string | null;
  flags: string | null;
  score: number;
  trim_start: number | null;
  trim_end: number | null;
  n_frames: number;
  num_frames: number | null;
  act_span: number | null;
  progress_dip: number | null;
  cluster: number | null;
  outlier_p: number | null;
  ax_poor_image: number | null;
  ax_jittery: number | null;
  ax_unsteady_camera: number | null;
  ax_hesitant: number | null;
  ax_incomplete: number | null;
  ax_combined: number | null;
  has_clip: boolean;
};

export const DISQUALIFYING_Z = 1.25;

export const AXES = [
  "poor_image",
  "jittery",
  "unsteady_camera",
  "hesitant",
  "incomplete",
] as const;

export type Axis = (typeof AXES)[number];

export const AXIS_FIELD: Record<Axis, keyof Episode> = {
  poor_image: "ax_poor_image",
  jittery: "ax_jittery",
  unsteady_camera: "ax_unsteady_camera",
  hesitant: "ax_hesitant",
  incomplete: "ax_incomplete",
};

export const LABELS: Record<string, string> = {
  poor_image: "poor image",
  jittery: "jittery",
  unsteady_camera: "unsteady camera",
  hesitant: "hesitant",
  incomplete: "incomplete",
};

const AXIS_TIP_SUFFIX = " — higher is worse, above 1.25 disqualifies";

export const AXIS_TIPS: Record<Axis, string> = {
  poor_image: "Frames with little image detail, faint frames, unstable compressed size" + AXIS_TIP_SUFFIX,
  jittery: "Low SPARC — hand and head motion is spectrally rough" + AXIS_TIP_SUFFIX,
  unsteady_camera: "Head jerk and head idle time" + AXIS_TIP_SUFFIX,
  hesitant: "Hands spend a large fraction of the clip stationary" + AXIS_TIP_SUFFIX,
  incomplete: "Short motion-energy span, or a wandering rather than direct path" + AXIS_TIP_SUFFIX,
};

export const FIELD_TIPS = {
  score: "Weighted composite, z-scored within operator",
  trim: "Motion-energy span: hand speed above 15% of this episode's peak",
  act_span: "Fraction of the clip containing movement",
  progress_dip:
    "Similarity drop from first frame to the most distant frame. Near zero = nothing visibly happened",
  outlier_p: "Calibrated p-value against this episode's own behaviour cluster",
  cluster: "Behaviour mode from k-means over episode descriptors",
  operator:
    "Demonstrator. Scores are normalised within operator so nobody is judged against another's baseline",
  reason: "Primary cause recorded by the pipeline for this verdict",
  flags: "Secondary warnings raised alongside the verdict",
  frames: "Frames present in the storage shard, and the count the catalog claims",
} as const;

export async function fetchEpisodes(): Promise<Episode[]> {
  const response = await fetch(`${CLIPS_API}/manifest.json`);
  if (!response.ok) throw new Error(`manifest unavailable (HTTP ${response.status})`);
  const data = await response.json();
  if (!Array.isArray(data)) throw new Error("manifest is not an array");
  return data as Episode[];
}

export function summarise(episodes: Episode[]) {
  const kept = episodes.filter((e) => e.decision === "keep").length;
  const defect = episodes.filter((e) => (e.reason ?? "") !== "").length;
  const total = episodes.length;
  return {
    total,
    kept,
    dropped: total - kept,
    keepPct: total ? (kept / total) * 100 : 0,
    dropPct: total ? ((total - kept) / total) * 100 : 0,
    defect,
    operators: new Set(episodes.map((e) => e.operator)).size,
  };
}

export function reasons(episodes: Episode[]) {
  const tally = new Map<string, number>();
  for (const e of episodes) {
    if (!e.reason) continue;
    tally.set(e.reason, (tally.get(e.reason) ?? 0) + 1);
  }
  return [...tally.entries()].sort((a, b) => b[1] - a[1]);
}

export function operatorList(episodes: Episode[]) {
  return [...new Set(episodes.map((e) => e.operator))].sort();
}

export function parseFlags(flags: string | null): string[] {
  return flags ? flags.split("+").filter(Boolean) : [];
}

export const num = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);
