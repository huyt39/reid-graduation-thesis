"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useAggregate } from "@/hooks/use-aggregate";
import { cn } from "@/lib/utils";

export function ActivityByDevice() {
  const { data, isLoading, error } = useAggregate({ group_by: "device" });

  if (isLoading && !data) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-muted-foreground">Sightings by Device</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-6" />
          ))}
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-muted-foreground">Sightings by Device</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-destructive">Failed to load activity data.</div>
        </CardContent>
      </Card>
    );
  }

  const buckets = (data ?? [])
    .slice()
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  const max = buckets[0]?.count ?? 1;

  if (buckets.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-muted-foreground">Sightings by Device</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
            No sighting data yet.
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-muted-foreground">Sightings by Device</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {buckets.map((bucket) => {
          const pct = Math.round((bucket.count / max) * 100);
          return (
            <div key={bucket._id} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="truncate text-sm">{bucket._id}</span>
                <span className="ml-4 shrink-0 tabular-nums text-muted-foreground">
                  {bucket.count.toLocaleString()}
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                <div
                  className={cn("h-full rounded-full bg-primary transition-all duration-300")}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
