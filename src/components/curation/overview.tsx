import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-3xl tabular-nums">{value}</CardTitle>
      </CardHeader>
      {hint ? (
        <CardContent className="pt-0 text-sm text-muted-foreground">{hint}</CardContent>
      ) : null}
    </Card>
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
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Episodes scored" value={stats.total.toLocaleString()} />
        <Stat label="Kept" value={stats.kept.toLocaleString()} />
        <Stat
          label="Dropped"
          value={stats.dropped.toLocaleString()}
          hint={`${stats.integrity} for measurable defects, ${stats.quota.toLocaleString()} by quota`}
        />
        <Stat label="Keep rate" value={`${(stats.keepRate * 100).toFixed(1)}%`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Integrity drops</CardTitle>
            <CardDescription>
              The {stats.integrity} episodes rejected for a defect in the data itself. Each rule is
              independently verifiable against the episode.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{ count: { label: "Episodes", color: "var(--chart-1)" } }}
              className="h-56 w-full"
            >
              <BarChart data={integrity} layout="vertical" margin={{ left: 8, right: 24 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={false} />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={150}
                  tickLine={false}
                  axisLine={false}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="count" fill="var(--color-count)" radius={4} />
              </BarChart>
            </ChartContainer>
            <dl className="mt-4 space-y-2 text-sm">
              {integrity.map((entry) => (
                <div key={entry.reason}>
                  <dt className="font-medium">{entry.label}</dt>
                  <dd className="text-muted-foreground">{REASON_EXPLANATIONS[entry.reason]}</dd>
                </div>
              ))}
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Keep rate by operator</CardTitle>
            <CardDescription>
              Scores are normalised within each operator and the quota is applied per operator, so
              nobody is disproportionately deleted. The line is the {KEEP_TARGET}% target; the
              spread below it is integrity drops.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{ keepRate: { label: "Keep rate %", color: "var(--chart-2)" } }}
              className="h-72 w-full"
            >
              <BarChart data={operatorRates} margin={{ left: 0, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="operator" tickLine={false} axisLine={false} hide />
                <YAxis domain={[0, 100]} tickLine={false} axisLine={false} width={36} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ReferenceLine
                  y={KEEP_TARGET}
                  stroke="var(--muted-foreground)"
                  strokeDasharray="4 4"
                />
                <Bar dataKey="keepRate" fill="var(--color-keepRate)" radius={4} />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Score distribution</CardTitle>
          <CardDescription>
            Kept and dropped episodes overlap because the cut is applied within each operator, not
            against a global threshold.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{
              keep: { label: "Keep", color: "var(--chart-2)" },
              drop: { label: "Drop", color: "var(--chart-5)" },
            }}
            className="h-64 w-full"
          >
            <BarChart data={histogram} margin={{ left: 0, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="score" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} width={40} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="drop" stackId="a" fill="var(--color-drop)" />
              <Bar dataKey="keep" stackId="a" fill="var(--color-keep)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  );
}
