"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useDevices } from "@/hooks/use-devices";
import { formatRelative } from "@/lib/date-format";

export function DeviceStatus() {
  const { data, isLoading, error } = useDevices();

  if (isLoading && !data) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
    );
  }

  if (error) {
    return <div className="text-sm text-destructive">Failed to load devices: {error.message}</div>;
  }

  const devices = data?.devices ?? [];

  if (devices.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-12 text-center text-sm text-muted-foreground">
        No devices registered.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {devices.map((d) => (
        <Link key={d.device_id} href={`/devices/${d.device_id}`}>
          <Card className="hover:border-primary/40 transition-colors h-full">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between text-sm">
                <span className="truncate">{d.device_id}</span>
                <Badge variant={d.status === "online" ? "default" : "secondary"} className="ml-2 shrink-0">
                  {d.status || "unknown"}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sightings</span>
                <span>{d.sighting_count.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Last frame</span>
                <span>{formatRelative(d.last_frame_at)}</span>
              </div>
              {d.location ? (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Location</span>
                  <span className="truncate ml-2 text-right">{d.location}</span>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
}
