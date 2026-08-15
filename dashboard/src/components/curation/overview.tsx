import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";

import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import {
  countFlags,
  countIntegrity,
  keepRateByOperator,
  scoreHistogram,
  summarise,
  AXIS_FLAG,
  AXIS_METHOD,
  AXIS_NOTES,
  REASON_EXPLANATIONS,
  type Episode,
} from "@/lib/curation";

const AXIS = {
  tickLine: false,
  axisLine: false,
  tick: { fontSize: 10, fontFamily: "var(--font-mono, ui-monospace)" },
} as const;

function Panel({
  title,
  note,
  children,
}: {
  title: string;
  note?: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="flex items-baseline gap-3 border-b border-border px-6 py-2.5">
        <h2 className="shrink-0 text-[11px] font-medium uppercase tracking-[0.2em]">{title}</h2>
        {note ? <p className="text-xs text-muted-foreground">{note}</p> : null}
      </div>
      <div className="px-6 py-6">{children}</div>
    </section>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="px-6 py-5">
      <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{label}</p>
      <p className="mt-2 font-mono text-3xl tabular-nums">{value}</p>
      <p className="mt-1 h-4 font-mono text-[11px] text-muted-foreground">{hint ?? ""}</p>
    </div>
  );
}

function ReasonList({
  entries,
  notes,
}: {
  entries: { reason: string; label: string }[];
  notes?: Record<string, string>;
}) {
  return (
    <dl className="mt-6 divide-y divide-border border-t border-border">
      {entries.map((entry) => (
        <div key={entry.reason} className="grid gap-1 py-2.5 sm:grid-cols-[13rem_1fr]">
          <dt className="font-mono text-[11px] uppercase tracking-wider">{entry.reason}</dt>
          <dd className="text-xs text-muted-foreground">
            {REASON_EXPLANATIONS[entry.reason]}
            {notes?.[entry.reason] ? (
              <span className="mt-1 block font-mono text-[10px] uppercase tracking-wider opacity-60">
                measured from: {notes[entry.reason]}
              </span>
            ) : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Overview({ episodes }: { episodes: Episode[] }) {
  const stats = summarise(episodes);
  const integrity = countIntegrity(episodes);
  const flags = countFlags(episodes);
  const operatorRates = keepRateByOperator(episodes);
  const histogram = scoreHistogram(episodes);
  const testedPercent = stats.total ? (stats.tested / stats.total) * 100 : 0;

  return (
    <div>
      <div className="grid grid-cols-2 divide-x divide-border border-b border-border lg:grid-cols-5">
        <Stat label="Episodes" value={stats.total.toLocaleString()} hint="scored" />
        <Stat
          label="Kept"
          value={stats.kept.toLocaleString()}
          hint={`${(stats.keepRate * 100).toFixed(1)}% of the slice`}
        />
        <Stat
          label="Integrity drops"
          value={stats.integrity.toLocaleString()}
          hint="verifiable defects"
        />
        <Stat
          label="Outlier drops"
          value={stats.outlier.toLocaleString()}
          hint={`of ${stats.tested.toLocaleString()} tested`}
        />
        <Stat
          label="Warning flags"
          value={stats.flagged.toLocaleString()}
          hint={`${stats.flaggedKeeps.toLocaleString()} of them kept`}
        />
      </div>

      <div className="border-b border-border bg-accent/30 px-6 py-3">
        <p className="font-mono text-[11px] uppercase tracking-wider text-chart-3">
          coverage · the outlier test examined {stats.tested.toLocaleString()} of{" "}
          {stats.total.toLocaleString()} episodes ({testedPercent.toFixed(1)}%)
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          The remaining {stats.untested.toLocaleString()} have no frame embeddings yet, so they were
          never tested for outlierness. They appear as kept because nothing rejected them — not
          because anything cleared them. Only the integrity rules and the warning flags cover the
          whole slice.
        </p>
      </div>

      <div className="grid border-b border-border lg:grid-cols-2 lg:divide-x lg:divide-border">
        <Panel
          title="Integrity drops"
          note={`${stats.integrity} episodes rejected for a defect in the data itself`}
        >
          <ChartContainer
            config={{ count: { label: "Episodes", color: "var(--chart-5)" } }}
            className="h-44 w-full"
          >
            <BarChart data={integrity} layout="vertical" margin={{ left: 8, right: 24 }}>
              <CartesianGrid horizontal={false} stroke="var(--border)" />
              <XAxis type="number" {...AXIS} />
              <YAxis type="category" dataKey="label" width={150} {...AXIS} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="count" fill="var(--color-count)" />
            </BarChart>
          </ChartContainer>
          <p className="mt-3 font-mono text-[11px] text-muted-foreground">
            each rule is independently checkable against the episode — these are claims about the
            data, not about taste
          </p>
          <ReasonList entries={integrity} />
        </Panel>

        <Panel
          title="Warning flags"
          note={`${stats.flagged.toLocaleString()} episodes cross a quality threshold without being rejected`}
        >
          <ChartContainer
            config={{ count: { label: "Episodes", color: "var(--chart-3)" } }}
            className="h-44 w-full"
          >
            <BarChart data={flags} layout="vertical" margin={{ left: 8, right: 24 }}>
              <CartesianGrid horizontal={false} stroke="var(--border)" />
              <XAxis type="number" {...AXIS} />
              <YAxis type="category" dataKey="label" width={150} {...AXIS} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="count" fill="var(--color-count)" />
            </BarChart>
          </ChartContainer>
          <p className="mt-3 font-mono text-[11px] text-muted-foreground">
            counted by containment · one episode can carry several · pairwise axis correlation ≤
            0.14, so these are distinct modes
          </p>
          <p className="mt-2 text-xs text-muted-foreground">{AXIS_METHOD}</p>
          <ReasonList entries={flags} notes={AXIS_NOTES} />
        </Panel>
      </div>

      <div className="grid border-b border-border lg:grid-cols-2 lg:divide-x lg:divide-border">
        <Panel
          title="Keep rate by operator"
          note="nothing constrains this — a wide spread would mean the engine is detecting demonstrators"
        >
          <ChartContainer
            config={{ keepRate: { label: "Keep rate %", color: "var(--chart-1)" } }}
            className="h-64 w-full"
          >
            <BarChart data={operatorRates} margin={{ left: 0, right: 8 }}>
              <CartesianGrid vertical={false} stroke="var(--border)" />
              <XAxis dataKey="operator" hide />
              <YAxis domain={[0, 100]} width={32} {...AXIS} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="keepRate" fill="var(--color-keepRate)" />
            </BarChart>
          </ChartContainer>
          <p className="mt-4 text-xs text-muted-foreground">
            One bar per demonstrator. Features whose variance is more than 35% explained by operator
            identity are excluded from scoring outright, and everything left is re-centred within
            each person — so a flat profile here is the intended result, not a coincidence.
          </p>
        </Panel>

        <Panel title="Score distribution" note="the composite quality score, stacked by decision">
          <ChartContainer
            config={{
              keep: { label: "Keep", color: "var(--chart-1)" },
              drop: { label: "Drop", color: "var(--chart-5)" },
            }}
            className="h-64 w-full"
          >
            <BarChart data={histogram} margin={{ left: 0, right: 8 }}>
              <CartesianGrid vertical={false} stroke="var(--border)" />
              <XAxis dataKey="score" {...AXIS} />
              <YAxis width={40} {...AXIS} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="drop" stackId="a" fill="var(--color-drop)" />
              <Bar dataKey="keep" stackId="a" fill="var(--color-keep)" />
            </BarChart>
          </ChartContainer>
          <p className="mt-4 text-xs text-muted-foreground">
            Each episode's features are re-centred on its own operator's median and averaged, so 0
            means typical and the axis reads in standard deviations. Drops are scattered rather than
            bunched at the bottom, because a broken episode is not the same thing as a low-scoring
            one — the integrity rules fire regardless of score. Flags cross at{" "}
            {AXIS_FLAG.toFixed(1)}.
          </p>
        </Panel>
      </div>

      <div className="border-b border-border px-6 py-6">
        <h2 className="text-[11px] font-medium uppercase tracking-[0.2em]">
          Where a decision comes from
        </h2>
        <ol className="mt-4 grid gap-px border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
          {[
            [
              "1 · Read the index",
              "One 16 KB ranged read per episode gives the compressed size of every frame, with no pixels downloaded. Faint and low-detail frames, unstable encoding, and frame-count disagreements all fall out of this.",
            ],
            [
              "2 · Read the poses",
              "Hand and head position over time gives smoothness (SPARC), jerk, idle fraction, reach straightness, and the span of the episode that actually contains movement.",
            ],
            [
              "3 · Look at the frames",
              "32 frames per episode through a vision model on a GPU, weighted toward the active span. Comparing each frame to the last one measures how much visibly happened.",
            ],
            [
              "4 · Decide",
              `Integrity rules reject outright. Everything else is re-centred per operator, grouped by embedding, and rejected only if its distance from its own group beats chance. Axes past ${AXIS_FLAG.toFixed(1)} raise a flag either way.`,
            ],
          ].map(([title, body]) => (
            <li key={title} className="bg-background px-4 py-4">
              <p className="font-mono text-[11px] uppercase tracking-wider text-primary">{title}</p>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{body}</p>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
