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
  countByReason,
  operators,
  REASON_EXPLANATIONS,
  REASON_LABELS,
  type Episode,
} from "@/lib/curation";

const PAGE_SIZE = 60;
const ALL = "all";
const APPROVED = "approved";
const REJECTED = "rejected";

function formatMetric(value: number | null) {
  return value === null || value === undefined ? "—" : value.toFixed(4);
}

function ClipDialog({ episode, onClose }: { episode: Episode | null; onClose: () => void }) {
  if (!episode) return null;

  const metrics: [string, string][] = [
    ["score", episode.score.toFixed(4)],
    ["operator", episode.operator.slice(0, 12)],
    ["frames", String(episode.n_frames)],
    ["catalog_frames", episode.num_frames ? String(Math.round(episode.num_frames)) : "—"],
    ["act_span", formatMetric(episode.act_span)],
    ["progress_dip", formatMetric(episode.progress_dip)],
    ["low_detail_frac", formatMetric(episode.low_detail_frac)],
    ["right_sparc", formatMetric(episode.right_sparc)],
    [
      "trim_span",
      episode.trim_start !== null && episode.trim_end !== null
        ? `${episode.trim_start} – ${episode.trim_end}`
        : "—",
    ],
  ];

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl gap-0 p-0">
        <DialogHeader className="space-y-1 border-b border-border px-6 py-3 text-left">
          <DialogTitle className="font-mono text-sm">{episode.episode_hash}</DialogTitle>
          <DialogDescription className="text-xs">
            <span
              className={
                episode.decision === "drop"
                  ? "font-mono uppercase tracking-widest text-destructive"
                  : "font-mono uppercase tracking-widest text-primary"
              }
            >
              {episode.decision === "drop" ? (episode.reason ?? "drop") : "keep"}
            </span>{" "}
            —{" "}
            {episode.decision === "drop"
              ? (REASON_EXPLANATIONS[episode.reason] ?? "")
              : "no defect found, and it scored above its operator's quota."}
          </DialogDescription>
        </DialogHeader>

        <div className="grid md:grid-cols-[1.5fr_1fr] md:divide-x md:divide-border">
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

          <dl className="divide-y divide-border">
            {metrics.map(([label, value]) => (
              <div key={label} className="flex items-baseline justify-between gap-4 px-6 py-2">
                <dt className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                  {label}
                </dt>
                <dd className="font-mono text-xs tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
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

  const reasonOptions = useMemo(() => countByReason(episodes), [episodes]);
  const operatorOptions = useMemo(() => operators(episodes), [episodes]);
  const approvedCount = useMemo(
    () => episodes.filter((e) => e.decision === "keep").length,
    [episodes],
  );

  const filtered = useMemo(() => {
    return episodes
      .filter((episode) => {
        if (operator !== ALL && episode.operator !== operator) return false;
        if (status === ALL) return true;
        if (status === APPROVED) return episode.decision === "keep";
        if (status === REJECTED) return episode.decision === "drop";
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
              <SelectItem value={APPROVED}>Approved ({approvedCount.toLocaleString()})</SelectItem>
              <SelectItem value={REJECTED}>
                Rejected ({(episodes.length - approvedCount).toLocaleString()})
              </SelectItem>
              <SelectGroup>
                <SelectLabel className="text-[11px] uppercase tracking-[0.2em]">
                  Rejected for
                </SelectLabel>
                {reasonOptions.map((option) => (
                  <SelectItem key={option.reason} value={option.reason}>
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
                    dropped ? "text-destructive" : "text-primary"
                  }`}
                >
                  {dropped ? (REASON_LABELS[episode.reason] ?? episode.reason) : "keep"}
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
