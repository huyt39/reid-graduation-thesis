"use client";

import Link from "next/link";
import { use } from "react";
import { ChevronLeft } from "lucide-react";
import { DashboardLayout } from "@/components/dashboard-layout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useDevice } from "@/hooks/use-devices";
import { formatDateTime } from "@/lib/date-format";

interface PageProps {
  params: Promise<{ deviceId: string }>;
}

export default function DeviceDetailPage({ params }: PageProps) {
  const { deviceId } = use(params);
  const { data: device, isLoading, error } = useDevice(deviceId);

  return (
    <DashboardLayout>
      <div className="space-y-4 p-4 md:p-6">
        <Link
          href="/devices"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          Back to devices
        </Link>

        {isLoading && !device ? (
          <Skeleton className="h-48 w-full" />
        ) : error || !device ? (
          <p className="text-destructive text-sm">
            Could not load device {deviceId}: {error?.message ?? "Not found"}
          </p>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="font-mono text-base">{device.device_id}</span>
                <Badge variant={device.status === "online" ? "default" : "secondary"}>
                  {device.status || "unknown"}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 text-sm">
              <Field label="Name" value={device.name || "—"} />
              <Field label="Location" value={device.location || "—"} />
              <Field
                label="Sightings"
                value={device.sighting_count.toLocaleString()}
              />
              <Field
                label="Unique persons"
                value={device.unique_person_count.toLocaleString()}
              />
              <Field label="First seen" value={formatDateTime(device.first_seen_at)} />
              <Field label="Last seen" value={formatDateTime(device.last_seen_at)} />
              <Field label="Last frame" value={formatDateTime(device.last_frame_at)} />
            </CardContent>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground uppercase tracking-wide">{label}</div>
      <div className="text-sm">{value}</div>
    </div>
  );
}
