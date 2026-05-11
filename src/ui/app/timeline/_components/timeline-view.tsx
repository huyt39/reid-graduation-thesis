"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { usePersonTimeline } from "@/hooks/use-person";
import { formatDateTime } from "@/lib/date-format";

export function TimelineView() {
  const [draftId, setDraftId] = useState("");
  const [personId, setPersonId] = useState<number | null>(null);
  const { data, isLoading, error } = usePersonTimeline(personId);

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const parsed = Number(draftId);
    setPersonId(Number.isFinite(parsed) && parsed > 0 ? parsed : null);
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Pick a person</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="flex items-end gap-3 max-w-md">
            <div className="flex-1 grid gap-2">
              <Label htmlFor="person_id">Person ID</Label>
              <Input
                id="person_id"
                type="number"
                placeholder="e.g. 42"
                value={draftId}
                onChange={(e) => setDraftId(e.target.value)}
              />
            </div>
            <Button type="submit">Load</Button>
          </form>
        </CardContent>
      </Card>

      {personId === null ? (
        <p className="text-sm text-muted-foreground rounded-lg border border-dashed p-12 text-center">
          Enter a person ID above to load their timeline.
        </p>
      ) : isLoading && !data ? (
        <Skeleton className="h-64 w-full" />
      ) : error ? (
        <p className="text-destructive text-sm">{error.message}</p>
      ) : (data?.items ?? []).length === 0 ? (
        <p className="text-sm text-muted-foreground rounded-lg border border-dashed p-12 text-center">
          No timeline events for person #{personId}.
        </p>
      ) : (
        <Card>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Event</TableHead>
                  <TableHead>Device</TableHead>
                  <TableHead>Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data!.items.map((ev, i) => (
                  <TableRow key={`${ev.timestamp}-${i}`}>
                    <TableCell>{formatDateTime(ev.timestamp)}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{ev.event_type}</Badge>
                    </TableCell>
                    <TableCell className="font-mono text-xs">{ev.device_id}</TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {Object.keys(ev.details ?? {}).length > 0
                        ? JSON.stringify(ev.details)
                        : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
