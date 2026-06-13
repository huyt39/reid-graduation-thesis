"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useDevices } from "@/hooks/use-devices";
import { getDeviceDisplayLocation, getDeviceDisplayName } from "@/lib/device-labels";

export function DevicesList() {
  const { data, isLoading, error } = useDevices();

  if (isLoading && !data) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-36" />
        ))}
      </div>
    );
  }

  if (error) {
    return <p className="text-destructive text-sm">{error.message}</p>;
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
            <CardHeader className="pb-3">
              <CardTitle className="text-base">
                <span className="text-sm">{d.device_id}</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Name</span>
                <span>{getDeviceDisplayName(d)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Location</span>
                <span>{getDeviceDisplayLocation()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sightings</span>
                <span>{d.sighting_count.toLocaleString()}</span>
              </div>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
}
