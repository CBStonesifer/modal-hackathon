import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";

import { ClipBrowser } from "@/components/curation/clip-browser";
import { Overview } from "@/components/curation/overview";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fetchEpisodes, summarise } from "@/lib/curation";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["episodes"],
    queryFn: fetchEpisodes,
    staleTime: Infinity,
  });

  const stats = data ? summarise(data) : null;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="border-b border-border">
        <div className="flex flex-wrap items-baseline justify-between gap-4 px-6 py-3">
          <div className="flex items-baseline gap-4">
            <span className="text-sm font-semibold uppercase tracking-[0.25em]">
              Curation Review
            </span>
            <span className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
              egoverse / mecka / cup_on_saucer
            </span>
          </div>
          <span className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
            {stats
              ? `${stats.total.toLocaleString()} episodes · ${(stats.keepRate * 100).toFixed(1)}% keep · ${stats.integrity} integrity drops · ${stats.flagged.toLocaleString()} flagged`
              : "loading manifest"}
          </span>
        </div>
      </header>

      {isLoading ? (
        <div className="space-y-px p-6">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-72 w-full" />
        </div>
      ) : error ? (
        <p className="px-6 py-8 font-mono text-sm text-destructive">
          manifest unavailable — {(error as Error).message}
        </p>
      ) : data ? (
        <Tabs defaultValue="overview">
          <TabsList className="h-auto w-full justify-start rounded-none border-b border-border bg-transparent p-0">
            {[
              ["overview", "Overview"],
              ["clips", "Clips"],
            ].map(([value, label]) => (
              <TabsTrigger
                key={value}
                value={value}
                className="rounded-none border-r border-border px-6 py-2.5 text-[11px] font-medium uppercase tracking-[0.2em] text-muted-foreground data-[state=active]:bg-accent data-[state=active]:text-foreground data-[state=active]:shadow-none"
              >
                {label}
              </TabsTrigger>
            ))}
          </TabsList>
          <TabsContent value="overview" className="mt-0">
            <Overview episodes={data} />
          </TabsContent>
          <TabsContent value="clips" className="mt-0">
            <ClipBrowser episodes={data} />
          </TabsContent>
        </Tabs>
      ) : null}

      <footer className="mt-auto flex flex-wrap justify-between gap-4 border-t border-border px-6 py-3 font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
        <span>source: keep_drop.csv · clips served from modal volume egoverse-footage</span>
        <span>dinov2-base · l4 · tiers 1–3</span>
      </footer>
    </div>
  );
}
