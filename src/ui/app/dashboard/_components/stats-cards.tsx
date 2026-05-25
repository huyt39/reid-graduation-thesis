"use client";

import { Users, UserCheck, Activity, Camera } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useStats } from "@/hooks/use-stats";

interface StatItem {
  label: string;
  value: number | undefined;
  icon: React.ComponentType<{ className?: string }>;
}

export function StatsCards() {
  const { data, isLoading, error } = useStats();

  const items: StatItem[] = [
    { label: "Total Persons", value: data?.total_persons, icon: Users },
    { label: "Active Persons", value: data?.active_persons, icon: UserCheck },
    { label: "Total Sightings", value: data?.total_sightings, icon: Activity },
    { label: "Devices", value: data?.total_devices, icon: Camera },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <Card key={item.label}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between">
                <span>{item.label}</span>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-8 w-24" />
              ) : error ? (
                <span className="text-destructive text-sm">—</span>
              ) : (
                <span className="text-3xl font-semibold tracking-tight">
                  {item.value?.toLocaleString() ?? 0}
                </span>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
