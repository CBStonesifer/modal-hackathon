import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts";

import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import {
  countByReason,
  keepRateByOperator,
  scoreHistogram,
  summarise,
  INTEGRITY_REASONS,
  REASON_EXPLANATIONS,
  type Episode,
} from "@/lib/curation";

const KEEP_TARGET = 70;

const AXIS = {
  tickLine: false,
  axisLine: false,
  tick: { fontSize: 10, fontFamily: "var(--font-mono, ui-monospace)" },
} as const;

function Panel({
  title,
  note,
  className,
  children,
}: {
  title: string;
  note?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={className}>
      <div className="flex items-baseline gap-3 border-b border-border px-6 py-2.5">
        <h2 className="text-[11px] font-medium uppercase tracking-[0.2em]">{title}</h2>
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

export function Overview({ episodes }: { episodes: Episode[] }) {
  const stats = summarise(episodes);
  const reasons = countByReason(episodes);
  const integrity = reasons.filter((r) =>
    (INTEGRITY_REASONS as readonly string[]).includes(r.reason),
  );
  const operatorRates = keepRateByOperator(episodes);
  const histogram = scoreHistogram(episodes);

  return (
    <div>
      <div className="grid grid-cols-2 divide-x divide-border border-b border-border lg:grid-cols-4">
        <Stat label="Episodes" value={stats.total.toLocaleString()} hint="scored" />
        <Stat
          label="Kept"
          value={stats.kept.toLocaleString()}
          hint={`${(stats.keepRate * 100).toFixed(1)}% of slice`}
        />
        <Stat
          label="Dropped"
          value={stats.dropped.toLocaleString()}
          hint={`${stats.integrity} defect / ${stats.quota.toLocaleString()} quota`}
        />
        <Stat label="Target" value={`${KEEP_TARGET}%`} hint="keep rate, per operator" />
      </div>

      <div className="grid border-b border-border lg:grid-cols-2 lg:divide-x lg:divide-border">
        <Panel
          title="Integrity drops"
          note={`${stats.integrity} episodes rejected for a defect in the data itself`}
        >
          <ChartContainer
            config={{ count: { label: "Episodes", color: "var(--chart-5)" } }}
            className="h-48 w-full"
          >
            <BarChart data={integrity} layout="vertical" margin={{ left: 8, right: 24 }}>
              <CartesianGrid horizontal={false} stroke="var(--border)" />
              <XAxis type="number" {...AXIS} />
              <YAxis type="category" dataKey="label" width={150} {...AXIS} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="count" fill="var(--color-count)" />
            </BarChart>
          </ChartContainer>

          <dl className="mt-6 divide-y divide-border border-t border-border">
            {integrity.map((entry) => (
              <div key={entry.reason} className="grid gap-1 py-2.5 sm:grid-cols-[13rem_1fr]">
                <dt className="font-mono text-[11px] uppercase tracking-wider">{entry.reason}</dt>
                <dd className="text-xs text-muted-foreground">
                  {REASON_EXPLANATIONS[entry.reason]}
                </dd>
              </div>
            ))}
          </dl>
        </Panel>

        <Panel
          title="Keep rate by operator"
          note="normalised and quota'd within each operator, so nobody is disproportionately deleted"
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
              <ReferenceLine y={KEEP_TARGET} stroke="var(--chart-3)" strokeDasharray="3 3" />
              <Bar dataKey="keepRate" fill="var(--color-keepRate)" />
            </BarChart>
          </ChartContainer>
          <p className="mt-4 font-mono text-[11px] text-muted-foreground">
            dashed line = {KEEP_TARGET}% target · spread below it is integrity drops
          </p>
        </Panel>
      </div>

      <Panel
        title="Score distribution"
        note="keep and drop overlap: the cut is applied within each operator, not against a global threshold"
      >
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
      </Panel>
    </div>
  );
}
