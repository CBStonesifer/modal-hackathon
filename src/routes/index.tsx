import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";

import { ClipBrowser } from "@/components/curation/clip-browser";
import { Overview } from "@/components/curation/overview";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fetchEpisodes } from "@/lib/curation";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["episodes"],
    queryFn: fetchEpisodes,
    staleTime: Infinity,
  });

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="mx-auto max-w-7xl px-6 py-8">
          <h1 className="text-2xl font-semibold tracking-tight">Curation review</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            EgoVerse <span className="font-mono">mecka / cup_on_saucer</span> — every keep and drop
            decision, the metric that produced it, and the footage to check it against.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {isLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-28 w-full" />
            <Skeleton className="h-72 w-full" />
          </div>
        ) : error ? (
          <p className="text-sm text-destructive">
            Could not load the manifest: {(error as Error).message}
          </p>
        ) : data ? (
          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="clips">Rejected clips</TabsTrigger>
            </TabsList>
            <TabsContent value="overview">
              <Overview episodes={data} />
            </TabsContent>
            <TabsContent value="clips">
              <ClipBrowser episodes={data} />
            </TabsContent>
          </Tabs>
        ) : null}
      </main>
    </div>
  );
}
