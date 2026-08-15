import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  clipUrl,
  countFlags,
  countIntegrity,
  hasFlag,
  operators,
  parseFlags,
  AXIS_FIELD,
  AXIS_FLAG,
  AXIS_METHOD,
  AXIS_NOTES,
  METRIC_NOTES,
  QUALITY_AXES,
  REASON_EXPLANATIONS,
  REASON_LABELS,
  type Episode,
} from "@/lib/curation";

const PAGE_SIZE = 60;
const ALL = "all";
const APPROVED = "approved";
const REJECTED = "rejected";
const UNTESTED = "untested";
const FLAG_PREFIX = "flag:";

function formatMetric(value: number | null) {
  return value === null || value === undefined ? "—" : value.toFixed(4);
}

function shortReason(reason: string | null) {
  if (!reason) return "";
  const primary = reason.split("+")[0];
  return REASON_LABELS[primary] ?? primary;
}

function AxisBar({ axis, value }: { axis: string; value: number | null }) {
  const failed = value !== null && value > AXIS_FLAG;
  // axes are z-scores within operator, roughly -4..4; centre the bar on 0
  const offset = value === null ? 0 : Math.max(-4, Math.min(4, value));
  return (
    <div className="px-6 py-2.5">
      <div className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3">
        <span
          className={`font-mono text-[11px] uppercase tracking-wider ${
            failed ? "text-chart-3" : "text-muted-foreground"
          }`}
        >
          {axis}
        </span>
        <span className="relative block h-px w-full bg-border">
          <span className="absolute left-1/2 top-[-3px] h-[7px] w-px bg-border" />
          <span
            className={`absolute top-[-1px] h-[3px] ${failed ? "bg-chart-3" : "bg-primary"}`}
            style={{
              left: offset >= 0 ? "50%" : `${50 + (offset / 4) * 50}%`,
              width: `${(Math.abs(offset) / 4) * 50}%`,
            }}
          />
        </span>
        <span className="text-right font-mono text-[11px] tabular-nums">
          {value === null ? "—" : value.toFixed(2)}
        </span>
      </div>
      <p className="mt-1 text-[11px] leading-snug text-muted-foreground">{AXIS_NOTES[axis]}</p>
    </div>
  );
}

function ClipDialog({ episode, onClose }: { episode: Episode | null; onClose: () => void }) {
  if (!episode) return null;

  const dropped = episode.decision === "drop";
  const flags = parseFlags(episode.flags);
  const metrics: [string, string][] = [
    ["score", episode.score.toFixed(4)],
    ["operator", episode.operator.slice(0, 12)],
    ["frames", String(episode.n_frames)],
    ["catalog_frames", episode.num_frames ? String(Math.round(episode.num_frames)) : "—"],
    ["act_span", formatMetric(episode.act_span)],
    ["progress_dip", formatMetric(episode.progress_dip)],
    ["cluster", episode.cluster === null ? "not tested" : String(episode.cluster)],
    ["outlier_p", episode.outlier_p === null ? "not tested" : episode.outlier_p.toExponential(2)],
    [
      "trim_span",
      episode.trim_start !== null && episode.trim_end !== null
        ? `${episode.trim_start} – ${episode.trim_end}`
        : "—",
    ],
  ];

  const verdict = dropped
    ? (REASON_EXPLANATIONS[episode.reason?.split("+")[0] ?? ""] ?? "")
    : flags.length
      ? `Kept, but flagged: ${flags.join(", ")}. A flag is a warning, not a rejection.`
      : episode.cluster === null
        ? "Kept. No integrity rule fired — but this episode has no embedding, so it was never tested for outlierness."
        : "Kept. No integrity rule fired, no axis crossed its threshold, and it sits close enough to its own behaviour group.";

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-4xl gap-0 overflow-y-auto p-0">
        <DialogHeader className="space-y-1 border-b border-border px-6 py-3 text-left">
          <DialogTitle className="font-mono text-sm">{episode.episode_hash}</DialogTitle>
          <DialogDescription className="text-xs">
            <span
              className={`font-mono uppercase tracking-widest ${
                dropped ? "text-destructive" : "text-primary"
              }`}
            >
              {dropped ? (episode.reason ?? "drop") : "keep"}
            </span>{" "}
            — {verdict}
          </DialogDescription>
        </DialogHeader>

        <div className="grid md:grid-cols-[1.4fr_1fr] md:divide-x md:divide-border">
          <div className="p-4">
            {episode.has_clip ? (
              <video
                key={episode.episode_hash}
                className="w-full border border-border bg-black"
                src={clipUrl(episode.episode_hash)}
                controls
                autoPlay
                loop
                muted
              />
            ) : (
              <div className="flex aspect-video items-center justify-center border border-border font-mono text-xs text-muted-foreground">
                no preview clip in volume
              </div>
            )}
          </div>

          <div>
            <dl className="divide-y divide-border">
              {metrics.map(([label, value]) => (
                <div key={label} className="px-6 py-2.5">
                  <div className="flex items-baseline justify-between gap-4">
                    <dt className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                      {label}
                    </dt>
                    <dd className="font-mono text-xs tabular-nums">{value}</dd>
                  </div>
                  <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
                    {METRIC_NOTES[label]}
                  </p>
                </div>
              ))}
            </dl>

            <div className="border-y border-border px-6 py-2.5">
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                failure axes · &gt; {AXIS_FLAG.toFixed(1)} = flagged
              </p>
              <p className="mt-1.5 text-[11px] leading-snug text-muted-foreground">{AXIS_METHOD}</p>
            </div>
            <div className="divide-y divide-border">
              {QUALITY_AXES.map((axis) => (
                <AxisBar
                  key={axis}
                  axis={axis}
                  value={episode[AXIS_FIELD[axis]] as number | null}
                />
              ))}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function ClipBrowser({ episodes }: { episodes: Episode[] }) {
  const [status, setStatus] = useState(ALL);
  const [operator, setOperator] = useState(ALL);
  const [visible, setVisible] = useState(PAGE_SIZE);
  const [selected, setSelected] = useState<Episode | null>(null);

  const integrityOptions = useMemo(() => countIntegrity(episodes), [episodes]);
  const flagOptions = useMemo(() => countFlags(episodes), [episodes]);
  const operatorOptions = useMemo(() => operators(episodes), [episodes]);
  const counts = useMemo(() => {
    const kept = episodes.filter((e) => e.decision === "keep").length;
    return {
      kept,
      dropped: episodes.length - kept,
      untested: episodes.filter((e) => e.cluster === null).length,
    };
  }, [episodes]);

  const filtered = useMemo(() => {
    return episodes
      .filter((episode) => {
        if (operator !== ALL && episode.operator !== operator) return false;
        if (status === ALL) return true;
        if (status === APPROVED) return episode.decision === "keep";
        if (status === REJECTED) return episode.decision === "drop";
        if (status === UNTESTED) return episode.cluster === null;
        if (status.startsWith(FLAG_PREFIX)) {
          return hasFlag(episode, status.slice(FLAG_PREFIX.length));
        }
        return episode.reason === status;
      })
      .sort((a, b) => b.score - a.score);
  }, [episodes, status, operator]);

  const resetPaging = () => setVisible(PAGE_SIZE);
  const selectClass =
    "w-full rounded-none border-border bg-transparent font-mono text-xs uppercase tracking-wider sm:w-72";

  return (
    <div>
      <div className="flex flex-wrap items-end gap-x-8 gap-y-4 border-b border-border px-6 py-4">
        <div className="space-y-1.5">
          <Label
            htmlFor="status"
            className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground"
          >
            Status
          </Label>
          <Select
            value={status}
            onValueChange={(value) => {
              setStatus(value);
              resetPaging();
            }}
          >
            <SelectTrigger id="status" className={selectClass}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="rounded-none font-mono text-xs">
              <SelectItem value={ALL}>All clips ({episodes.length.toLocaleString()})</SelectItem>
              <SelectItem value={APPROVED}>Approved ({counts.kept.toLocaleString()})</SelectItem>
              <SelectItem value={REJECTED}>Rejected ({counts.dropped.toLocaleString()})</SelectItem>
              <SelectItem value={UNTESTED}>
                Never outlier-tested ({counts.untested.toLocaleString()})
              </SelectItem>
              <SelectGroup>
                <SelectLabel className="text-[11px] uppercase tracking-[0.2em]">
                  Rejected for
                </SelectLabel>
                {integrityOptions.map((option) => (
                  <SelectItem key={option.reason} value={option.reason}>
                    {option.label} ({option.count})
                  </SelectItem>
                ))}
              </SelectGroup>
              <SelectGroup>
                <SelectLabel className="text-[11px] uppercase tracking-[0.2em]">
                  Flagged for
                </SelectLabel>
                {flagOptions.map((option) => (
                  <SelectItem key={option.reason} value={`${FLAG_PREFIX}${option.reason}`}>
                    {option.label} ({option.count})
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label
            htmlFor="operator"
            className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground"
          >
            Operator
          </Label>
          <Select
            value={operator}
            onValueChange={(value) => {
              setOperator(value);
              resetPaging();
            }}
          >
            <SelectTrigger id="operator" className={`${selectClass} sm:w-56`}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="rounded-none font-mono text-xs">
              <SelectItem value={ALL}>All operators</SelectItem>
              {operatorOptions.map((value) => (
                <SelectItem key={value} value={value}>
                  {value.slice(0, 12)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <p className="pb-2 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
          {filtered.length.toLocaleString()} of {episodes.length.toLocaleString()} episodes
        </p>
      </div>

      <div className="grid border-b border-border sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.slice(0, visible).map((episode) => {
          const dropped = episode.decision === "drop";
          const flags = parseFlags(episode.flags);
          return (
            <button
              key={episode.episode_hash}
              type="button"
              onClick={() => setSelected(episode)}
              className="group -mb-px -mr-px border-b border-r border-border px-4 py-3 text-left transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none"
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-xs text-muted-foreground group-hover:text-foreground">
                  {episode.episode_hash.slice(0, 12)}
                </span>
                <span
                  className={`font-mono text-[10px] uppercase tracking-widest ${
                    dropped ? "text-destructive" : flags.length ? "text-chart-3" : "text-primary"
                  }`}
                >
                  {dropped ? shortReason(episode.reason) : flags.length ? flags[0] : "keep"}
                </span>
              </div>
              <div className="mt-2 flex items-baseline justify-between font-mono text-[11px] text-muted-foreground">
                <span>score {episode.score.toFixed(3)}</span>
                <span>{episode.n_frames} frames</span>
              </div>
              <div className="mt-2 h-px w-full bg-border">
                <div
                  className={dropped ? "h-px bg-destructive" : "h-px bg-primary"}
                  style={{ width: `${Math.min(100, Math.max(2, (episode.score + 2) * 33))}%` }}
                />
              </div>
            </button>
          );
        })}
      </div>

      {visible < filtered.length ? (
        <div className="flex justify-center border-b border-border py-6">
          <Button
            variant="outline"
            onClick={() => setVisible((count) => count + PAGE_SIZE)}
            className="rounded-none font-mono text-[11px] uppercase tracking-[0.2em]"
          >
            Load {Math.min(PAGE_SIZE, filtered.length - visible)} more
          </Button>
        </div>
      ) : null}

      <ClipDialog episode={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
