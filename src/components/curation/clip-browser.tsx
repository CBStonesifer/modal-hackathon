import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  clipUrl,
  countByReason,
  operators,
  REASON_EXPLANATIONS,
  REASON_LABELS,
  type Episode,
} from "@/lib/curation";

const PAGE_SIZE = 48;
const ALL = "all";

function formatMetric(value: number | null) {
  return value === null || value === undefined ? "—" : value.toFixed(4);
}

function ClipDialog({ episode, onClose }: { episode: Episode | null; onClose: () => void }) {
  if (!episode) return null;

  const metrics: [string, string][] = [
    ["Score", episode.score.toFixed(4)],
    ["Frames", String(episode.n_frames)],
    ["Catalog frames", episode.num_frames ? String(Math.round(episode.num_frames)) : "—"],
    ["Active span", formatMetric(episode.act_span)],
    ["Progress dip", formatMetric(episode.progress_dip)],
    ["Low detail fraction", formatMetric(episode.low_detail_frac)],
    ["SPARC (right hand)", formatMetric(episode.right_sparc)],
    [
      "Trim span",
      episode.trim_start !== null && episode.trim_end !== null
        ? `${episode.trim_start} – ${episode.trim_end}`
        : "—",
    ],
  ];

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="font-mono text-base">{episode.episode_hash}</DialogTitle>
          <DialogDescription>
            {episode.decision === "drop"
              ? (REASON_EXPLANATIONS[episode.reason] ?? episode.reason)
              : "Kept — no defect found and it scored above its operator's quota."}
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 md:grid-cols-[1.4fr_1fr]">
          {episode.has_clip ? (
            <video
              key={episode.episode_hash}
              className="w-full rounded-lg bg-muted"
              src={clipUrl(episode.episode_hash)}
              controls
              autoPlay
              loop
              muted
            />
          ) : (
            <div className="flex aspect-video items-center justify-center rounded-lg bg-muted text-sm text-muted-foreground">
              No preview clip in the volume
            </div>
          )}

          <dl className="space-y-2 text-sm">
            <div className="flex items-baseline justify-between gap-4">
              <dt className="text-muted-foreground">Operator</dt>
              <dd className="font-mono text-xs">{episode.operator.slice(0, 12)}</dd>
            </div>
            {metrics.map(([label, value]) => (
              <div key={label} className="flex items-baseline justify-between gap-4">
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="tabular-nums">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function ClipBrowser({ episodes }: { episodes: Episode[] }) {
  const [reason, setReason] = useState(ALL);
  const [operator, setOperator] = useState(ALL);
  const [includeKeeps, setIncludeKeeps] = useState(false);
  const [visible, setVisible] = useState(PAGE_SIZE);
  const [selected, setSelected] = useState<Episode | null>(null);

  const reasonOptions = useMemo(() => countByReason(episodes), [episodes]);
  const operatorOptions = useMemo(() => operators(episodes), [episodes]);

  const filtered = useMemo(() => {
    return episodes.filter((episode) => {
      if (!includeKeeps && episode.decision !== "drop") return false;
      if (reason !== ALL && episode.reason !== reason) return false;
      if (operator !== ALL && episode.operator !== operator) return false;
      return true;
    });
  }, [episodes, includeKeeps, reason, operator]);

  const resetPaging = () => setVisible(PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1.5">
          <Label htmlFor="reason">Reason</Label>
          <Select
            value={reason}
            onValueChange={(value) => {
              setReason(value);
              resetPaging();
            }}
          >
            <SelectTrigger id="reason" className="w-64">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL}>All reasons</SelectItem>
              {reasonOptions.map((option) => (
                <SelectItem key={option.reason} value={option.reason}>
                  {option.label} ({option.count})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="operator">Operator</Label>
          <Select
            value={operator}
            onValueChange={(value) => {
              setOperator(value);
              resetPaging();
            }}
          >
            <SelectTrigger id="operator" className="w-56">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL}>All operators</SelectItem>
              {operatorOptions.map((value) => (
                <SelectItem key={value} value={value} className="font-mono text-xs">
                  {value.slice(0, 12)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2 pb-2">
          <Switch
            id="keeps"
            checked={includeKeeps}
            onCheckedChange={(checked) => {
              setIncludeKeeps(checked);
              resetPaging();
            }}
          />
          <Label htmlFor="keeps">Include kept episodes</Label>
        </div>

        <p className="pb-2 text-sm text-muted-foreground">
          {filtered.length.toLocaleString()} episodes
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.slice(0, visible).map((episode) => (
          <Card
            key={episode.episode_hash}
            role="button"
            tabIndex={0}
            onClick={() => setSelected(episode)}
            onKeyDown={(event) => event.key === "Enter" && setSelected(episode)}
            className="cursor-pointer transition-colors hover:border-ring focus-visible:border-ring focus-visible:outline-none"
          >
            <CardContent className="space-y-2 p-4">
              <div className="flex items-start justify-between gap-2">
                <span className="font-mono text-xs text-muted-foreground">
                  {episode.episode_hash.slice(0, 12)}
                </span>
                <Badge variant={episode.decision === "drop" ? "destructive" : "secondary"}>
                  {episode.decision === "drop"
                    ? (REASON_LABELS[episode.reason] ?? episode.reason)
                    : "keep"}
                </Badge>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">score</span>
                <span className="tabular-nums">{episode.score.toFixed(3)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">frames</span>
                <span className="tabular-nums">{episode.n_frames}</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {visible < filtered.length ? (
        <div className="flex justify-center">
          <Button variant="outline" onClick={() => setVisible((count) => count + PAGE_SIZE)}>
            Show more
          </Button>
        </div>
      ) : null}

      <ClipDialog episode={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
