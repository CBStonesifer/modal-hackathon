import { Tip } from "./tip";
import {
  AXES,
  AXIS_FIELD,
  AXIS_TIPS,
  DISQUALIFYING_Z,
  FIELD_TIPS,
  LABELS,
  clipUrl,
  num,
  parseFlags,
  type Episode,
} from "@/lib/curation";
import { cn } from "@/lib/utils";

function Row({ label, tip, children }: { label: string; tip?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-border/60 px-3 py-1.5">
      <span className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
        {tip ? <Tip tip={tip}>{label}</Tip> : label}
      </span>
      <span className="font-mono text-[12px] tabular-nums text-foreground">{children}</span>
    </div>
  );
}

function AxisBar({ axis, z }: { axis: (typeof AXES)[number]; z: number | null }) {
  const value = z ?? 0;
  const bad = value > DISQUALIFYING_Z;
  const pct = Math.max(0, Math.min(100, ((value + 4) / 8) * 100));
  return (
    <div className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3 px-3 py-1.5">
      <span
        className={cn(
          "text-[11px] uppercase tracking-[0.14em]",
          bad ? "font-bold text-destructive" : "text-muted-foreground",
        )}
      >
        <Tip tip={AXIS_TIPS[axis]}>{LABELS[axis]}</Tip>
      </span>
      <span className="relative block h-2 bg-muted">
        <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
        <span
          className={cn("absolute inset-y-0 left-0", bad ? "bg-destructive" : "bg-primary/70")}
          style={{ width: `${pct}%` }}
        />
      </span>
      <span
        className={cn(
          "text-right font-mono text-[12px] tabular-nums",
          bad ? "font-bold text-destructive" : "text-foreground",
        )}
      >
        {num(z, 2)}
      </span>
    </div>
  );
}

export function DetailPanel({ episode }: { episode: Episode | null }) {
  if (!episode) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
        select an episode
      </div>
    );
  }

  const total = episode.n_frames || episode.num_frames || 0;
  const span =
    episode.trim_start !== null && episode.trim_end !== null && total
      ? `frames ${episode.trim_start}–${episode.trim_end} of ${total} (${(
          ((episode.trim_end - episode.trim_start) / total) *
          100
        ).toFixed(0)}% of clip)`
      : "—";

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="flex items-baseline justify-between gap-3 border-b border-border px-3 py-2">
        <span className="font-mono text-[12px] text-foreground">{episode.episode_hash}</span>
        <span
          className={cn(
            "text-[11px] font-semibold uppercase tracking-[0.2em]",
            episode.decision === "keep" ? "text-keep" : "text-drop",
          )}
        >
          {episode.decision}
        </span>
      </div>

      <div className="border-b border-border bg-black">
        {episode.has_clip ? (
          <video
            key={episode.episode_hash}
            src={clipUrl(episode.episode_hash)}
            controls
            loop
            muted
            playsInline
            className="aspect-video w-full"
          />
        ) : (
          <div className="flex aspect-video w-full items-center justify-center text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
            no clip available
          </div>
        )}
      </div>

      <div>
        <Row label="reason" tip={FIELD_TIPS.reason}>
          {episode.reason ?? "—"}
        </Row>
        <Row label="flags" tip={FIELD_TIPS.flags}>
          {parseFlags(episode.flags).join(" · ") || "—"}
        </Row>
        <Row label="score" tip={FIELD_TIPS.score}>
          {num(episode.score, 3)}
        </Row>
        <Row label="trim span" tip={FIELD_TIPS.trim}>
          {span}
        </Row>
        <Row label="frames" tip={FIELD_TIPS.frames}>
          {episode.n_frames ?? "—"} / {episode.num_frames ?? "—"}
        </Row>
        <Row label="act span" tip={FIELD_TIPS.act_span}>
          {num(episode.act_span, 3)}
        </Row>
        <Row label="progress dip" tip={FIELD_TIPS.progress_dip}>
          {num(episode.progress_dip, 4)}
        </Row>
        <Row label="outlier p" tip={FIELD_TIPS.outlier_p}>
          {episode.outlier_p === null ? "—" : episode.outlier_p.toExponential(2)}
        </Row>
        <Row label="cluster" tip={FIELD_TIPS.cluster}>
          {episode.cluster ?? "—"}
        </Row>
        <Row label="operator" tip={FIELD_TIPS.operator}>
          {episode.operator}
        </Row>
      </div>

      <div className="mt-3 border-t border-border pt-2 pb-4">
        <div className="px-3 pb-1 text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          failure axes · z ∈ [-4, 4]
        </div>
        {AXES.map((axis) => (
          <AxisBar key={axis} axis={axis} z={episode[AXIS_FIELD[axis]] as number | null} />
        ))}
      </div>
    </div>
  );
}
