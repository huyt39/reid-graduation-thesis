"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PersonSnapshot } from "@/components/person-snapshot";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  usePerson,
  usePersonSightings,
  usePersonTimeline,
  usePersonSimilar,
} from "@/hooks/use-person";
import { formatDateTime, formatRelative } from "@/lib/date-format";

export function PersonDetail({ personId }: { personId: number }) {
  const { data: person, isLoading, error } = usePerson(personId);

  if (isLoading && !person) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (error || !person) {
    return (
      <div className="text-sm text-destructive">
        Could not load person #{personId}: {error?.message ?? "Not found"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Person #{person.person_id}</span>
            <Badge variant={person.is_active ? "default" : "secondary"}>
              {person.is_active ? "Active" : "Inactive"}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-6 lg:grid-cols-[240px_minmax(0,1fr)]">
          <PersonSnapshot
            src={person.snapshot_url}
            alt={`Person ${person.person_id} snapshot`}
            label={`#${person.person_id}`}
            className="aspect-[4/5] min-h-64"
          />
          <div className="grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
            <Field label="Gender" value={person.attributes.gender || "—"} />
            <Field
              label="Gender confidence"
              value={`${(person.attributes.gender_confidence * 100).toFixed(1)}%`}
            />
            <Field label="Sightings" value={person.stats.sighting_count.toLocaleString()} />
            <Field label="Last device" value={person.stats.last_seen_device || "—"} />
            <Field label="First seen" value={formatDateTime(person.stats.first_seen_at)} />
            <Field label="Last seen" value={formatDateTime(person.stats.last_seen_at)} />
            <Field label="Source" value={person.source} />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="sightings">
        <TabsList>
          <TabsTrigger value="sightings">Sightings</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
          <TabsTrigger value="similar">Similar</TabsTrigger>
        </TabsList>
        <TabsContent value="sightings">
          <SightingsTab personId={personId} />
        </TabsContent>
        <TabsContent value="timeline">
          <TimelineTab personId={personId} />
        </TabsContent>
        <TabsContent value="similar">
          <SimilarTab personId={personId} />
        </TabsContent>
      </Tabs>
    </div>
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

function SightingsTab({ personId }: { personId: number }) {
  const { data, isLoading, error } = usePersonSightings(personId);
  if (isLoading && !data) return <Skeleton className="h-48 w-full" />;
  if (error) return <p className="text-destructive text-sm">{error.message}</p>;
  const items = data?.items ?? [];
  if (items.length === 0) {
    return (
      <p className="text-sm text-muted-foreground rounded-lg border border-dashed p-8 text-center">
        No sightings recorded.
      </p>
    );
  }
  return (
    <Card>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Snapshot</TableHead>
              <TableHead>Device</TableHead>
              <TableHead>Started</TableHead>
              <TableHead>Ended</TableHead>
              <TableHead className="text-right">Duration (s)</TableHead>
              <TableHead className="text-right">Quality</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((s) => (
              <TableRow key={s.tracklet_id}>
                <TableCell>
                  <PersonSnapshot
                    src={s.snapshot_url}
                    alt={`Sighting ${s.tracklet_id} snapshot`}
                    label="Shot"
                    className="h-16 w-12 rounded-md"
                  />
                </TableCell>
                <TableCell className="text-xs">{s.device_id}</TableCell>
                <TableCell>{formatDateTime(s.started_at)}</TableCell>
                <TableCell>{formatDateTime(s.ended_at)}</TableCell>
                <TableCell className="text-right">{s.duration_seconds.toFixed(1)}</TableCell>
                <TableCell className="text-right">{(s.quality_score * 100).toFixed(0)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function TimelineTab({ personId }: { personId: number }) {
  const { data, isLoading, error } = usePersonTimeline(personId);
  if (isLoading && !data) return <Skeleton className="h-48 w-full" />;
  if (error) return <p className="text-destructive text-sm">{error.message}</p>;
  const items = data?.items ?? [];
  if (items.length === 0) {
    return (
      <p className="text-sm text-muted-foreground rounded-lg border border-dashed p-8 text-center">
        No timeline events.
      </p>
    );
  }
  return (
    <Card>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Time</TableHead>
              <TableHead>Event</TableHead>
              <TableHead>Device</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((ev, i) => (
              <TableRow key={`${ev.timestamp}-${i}`}>
                <TableCell>{formatDateTime(ev.timestamp)}</TableCell>
                <TableCell>
                  <Badge variant="outline">{ev.event_type}</Badge>
                </TableCell>
                <TableCell className="text-xs">{ev.device_id}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function SimilarTab({ personId }: { personId: number }) {
  const { data, isLoading, error } = usePersonSimilar(personId);
  if (isLoading && !data) return <Skeleton className="h-48 w-full" />;
  if (error) return <p className="text-destructive text-sm">{error.message}</p>;
  const items = data?.similar_persons ?? [];
  if (items.length === 0) {
    return (
      <p className="text-sm text-muted-foreground rounded-lg border border-dashed p-8 text-center">
        No similar persons found.
      </p>
    );
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      {items.map((s) => (
        <Link key={s.person_id} href={`/persons/${s.person_id}`}>
          <Card className="hover:border-primary/40 transition-colors">
            <CardContent className="flex items-center gap-3 p-4">
              <PersonSnapshot
                src={s.person?.snapshot_url}
                alt={`Person ${s.person_id} snapshot`}
                label={`#${s.person_id}`}
                className="h-20 w-16 shrink-0 rounded-md"
              />
              <div className="min-w-0 flex-1">
                <div className="font-medium">Person #{s.person_id}</div>
                {s.person && (
                  <div className="text-xs text-muted-foreground">
                    Last seen {formatRelative(s.person.stats.last_seen_at)}
                  </div>
                )}
              </div>
              <Badge variant="secondary">{(s.score * 100).toFixed(1)}%</Badge>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
}
