import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";

import { DetailPanel } from "@/components/curation/detail-panel";
import {
  fetchEpisodes,
  operatorList,
  reasons,
  summarise,
  type Episode,
} from "@/lib/curation";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "EgoVerse Curation Review — audit keep/drop verdicts" },
      {
        name: "description",
        content:
          "Inspect 5,407 first-person cup-on-saucer episodes: keep/drop verdicts, quality z-scores, and the footage behind each decision.",
      },
      { property: "og:title", content: "EgoVerse Curation Review" },
      {
        property: "og:description",
        content:
          "Audit automated keep/drop verdicts for a robot-learning dataset against the actual clips.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

const ROW_CAP = 300;

const selectClass =
  "h-7 border border-input bg-background px-2 font-mono text-[11px] text-foreground focus:outline-none focus:ring-1 focus:ring-ring";

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="border-l border-border px-4 first:border-l-0 first:pl-0">
      <div className="font-mono text-[22px] leading-none tabular-nums text-foreground">
        {value}
        {sub ? <span className="ml-1.5 text-[12px] text-muted-foreground">{sub}</span> : null}
      </div>
      <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
        {label}
      </div>
    </div>
  );
}

function Index() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["episodes"],
    queryFn: fetchEpisodes,
    staleTime: Infinity,
  });

  const [dark, setDark] = useState(true);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const [decision, setDecision] = useState("all");
  const [reason, setReason] = useState("all");
  const [operator, setOperator] = useState("all");
  const [sort, setSort] = useState("worst");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const episodes = useMemo(() => data ?? [], [data]);
  const stats = useMemo(() => summarise(episodes), [episodes]);
  const reasonOptions = useMemo(() => reasons(episodes), [episodes]);
  const operatorOptions = useMemo(() => operatorList(episodes), [episodes]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const rows = episodes.filter((e: Episode) => {
      if (decision !== "all" && e.decision !== decision) return false;
      if (reason !== "all" && (e.reason ?? "none") !== reason) return false;
      if (operator !== "all" && e.operator !== operator) return false;
      if (q && !e.episode_hash.toLowerCase().includes(q)) return false;
      return true;
    });
    const sorted = [...rows];
    if (sort === "best") sorted.sort((a, b) => b.score - a.score);
    else if (sort === "worst") sorted.sort((a, b) => a.score - b.score);
    else sorted.sort((a, b) => a.episode_hash.localeCompare(b.episode_hash));
    return sorted;
  }, [episodes, decision, reason, operator, sort, search]);

  const visible = filtered.slice(0, ROW_CAP);
  const overflow = filtered.length - visible.length;
  const current = useMemo(
    () => episodes.find((e) => e.episode_hash === selected) ?? visible[0] ?? null,
    [episodes, selected, visible],
  );

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <header className="shrink-0 border-b border-border">
        <div className="flex flex-wrap items-end justify-between gap-6 px-4 py-3">
          <div>
            <h1 className="text-[13px] font-semibold uppercase tracking-[0.24em]">
              EgoVerse Curation Review
            </h1>
            <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              cup_on_saucer · automated keep/drop audit
            </p>
          </div>
          <div className="flex items-end gap-0">
            <Tile label="episodes" value={stats.total.toLocaleString()} />
            <Tile
              label="kept"
              value={stats.kept.toLocaleString()}
              sub={`${stats.keepPct.toFixed(1)}%`}
            />
            <Tile
              label="dropped"
              value={stats.dropped.toLocaleString()}
              sub={`${stats.dropPct.toFixed(1)}%`}
            />
            <Tile label="named defect" value={stats.defect.toLocaleString()} />
            <Tile label="operators" value={String(stats.operators)} />
          </div>
          <button
            onClick={() => setDark((v) => !v)}
            className="h-7 border border-input px-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            {dark ? "light" : "dark"}
          </button>
        </div>
      </header>

      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <select
          className={selectClass}
          value={decision}
          onChange={(e) => setDecision(e.target.value)}
          aria-label="Decision"
        >
          <option value="all">decision: all</option>
          <option value="keep">keep</option>
          <option value="drop">drop</option>
        </select>
        <select
          className={selectClass}
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          aria-label="Reason"
        >
          <option value="all">reason: all</option>
          <option value="none">none</option>
          {reasonOptions.map(([r, count]) => (
            <option key={r} value={r}>
              {r} ({count})
            </option>
          ))}
        </select>
        <select
          className={selectClass}
          value={operator}
          onChange={(e) => setOperator(e.target.value)}
          aria-label="Operator"
        >
          <option value="all">operator: all</option>
          {operatorOptions.map((o) => (
            <option key={o} value={o}>
              {o.slice(0, 12)}
            </option>
          ))}
        </select>
        <select
          className={selectClass}
          value={sort}
          onChange={(e) => setSort(e.target.value)}
          aria-label="Sort"
        >
          <option value="worst">sort: worst score first</option>
          <option value="best">sort: best score first</option>
          <option value="hash">sort: hash</option>
        </select>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="search hash…"
          aria-label="Search hash"
          className={cn(selectClass, "w-52 placeholder:text-muted-foreground")}
        />
        <span className="ml-auto font-mono text-[11px] tabular-nums text-muted-foreground">
          {filtered.length.toLocaleString()} match
        </span>
      </div>

      {isLoading ? (
        <div className="p-4 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          loading manifest…
        </div>
      ) : error ? (
        <div className="p-4">
          <p className="font-mono text-[12px] text-destructive">
            manifest failed — {(error as Error).message}
          </p>
          <button
            onClick={() => refetch()}
            className="mt-3 h-7 border border-input px-3 text-[10px] uppercase tracking-[0.18em] hover:bg-accent"
          >
            retry
          </button>
        </div>
      ) : (
        <main className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
          <div className="min-h-0 overflow-y-auto border-b border-border md:border-b-0 md:border-r">
            {visible.map((e) => {
              const active = current?.episode_hash === e.episode_hash;
              return (
                <button
                  key={e.episode_hash}
                  onClick={() => setSelected(e.episode_hash)}
                  className={cn(
                    "flex w-full items-center gap-2 border-b border-border/60 px-3 py-1.5 text-left hover:bg-accent",
                    active && "bg-accent",
                  )}
                >
                  <span
                    className={cn(
                      "size-2 shrink-0 rounded-full",
                      e.decision === "keep" ? "bg-keep" : "bg-drop",
                    )}
                  />
                  <span className="font-mono text-[11px] text-foreground">
                    {e.episode_hash.slice(0, 12)}
                  </span>
                  <span className="ml-auto font-mono text-[11px] tabular-nums text-muted-foreground">
                    {e.score.toFixed(2)}
                  </span>
                  <span className="w-32 shrink-0 truncate text-right text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                    {e.reason ?? "—"}
                  </span>
                </button>
              );
            })}
            {overflow > 0 ? (
              <p className="px-3 py-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                {overflow.toLocaleString()} more match — narrow the filters
              </p>
            ) : null}
            {filtered.length === 0 ? (
              <p className="px-3 py-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                no episodes match
              </p>
            ) : null}
          </div>
          <div className="min-h-0 overflow-hidden">
            <DetailPanel episode={current} />
          </div>
        </main>
      )}
    </div>
  );
}
