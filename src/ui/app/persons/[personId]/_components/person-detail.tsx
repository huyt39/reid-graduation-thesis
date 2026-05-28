"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PersonSnapshot } from "@/components/person-snapshot";
import { TrackletEvidenceStrip } from "@/components/tracklet-evidence-strip";
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
  usePersonSimilar,
  usePersonTimeline,
  usePersonTracklets,
} from "@/hooks/use-person";
import {
  buildLiveEvidenceSummary,
  describeMatchMethod,
  formatDecimal,
  formatPct,
  getStatusClasses,
  getTrackletConfidenceLabel,
  summarizeTracklets,
} from "@/lib/reid-evidence";
import { cn } from "@/lib/utils";
import { formatDateTime, formatRelative } from "@/lib/date-format";
import { confidenceLabel, getAttributeGroups } from "@/lib/person-attributes";
import type { PersonAttributes, Tracklet } from "@/types";

export function PersonDetail({ personId }: { personId: number }) {
  const { data: person, isLoading, error } = usePerson(personId);
  const { data: trackletsData, isLoading: isTrackletsLoading } = usePersonTracklets(personId);

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

  const tracklets = trackletsData?.items ?? [];
  const trackletsById = new Map(tracklets.map((tracklet) => [tracklet.tracklet_id, tracklet]));
  const attributeGroups = getAttributeGroups(person.attributes);

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
            previewTitle={`Person #${person.person_id} snapshot`}
            previewDescription="Representative snapshot for this person profile."
          />
          <div className="space-y-5">
            <div className="grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
              <Field label="Sightings" value={person.stats.sighting_count.toLocaleString()} />
              <Field label="Last device" value={person.stats.last_seen_device || "—"} />
              <Field label="First seen" value={formatDateTime(person.stats.first_seen_at)} />
              <Field label="Last seen" value={formatDateTime(person.stats.last_seen_at)} />
              <Field label="Source" value={person.source} />
            </div>
            <AttributesPanel attributes={person.attributes} groups={attributeGroups} />
            <EvidenceSummary tracklets={tracklets} isLoading={isTrackletsLoading} />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="sightings">
        <TabsList>
          <TabsTrigger value="sightings">Sightings</TabsTrigger>
          <TabsTrigger value="evidence">Evidence</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
          <TabsTrigger value="similar">Similar</TabsTrigger>
        </TabsList>
        <TabsContent value="sightings">
          <SightingsTab personId={personId} trackletsById={trackletsById} />
        </TabsContent>
        <TabsContent value="evidence">
          <EvidenceTab tracklets={tracklets} isLoading={isTrackletsLoading} />
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

function AttributesPanel({
  attributes,
  groups,
}: {
  attributes: PersonAttributes;
  groups: ReturnType<typeof getAttributeGroups>;
}) {
  if (groups.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
        Attribute evidence is not confident enough to display yet.
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-muted/20 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-medium">Attributes</div>
          <div className="text-xs text-muted-foreground">
            Stable person-level votes from confirmed tracklets
          </div>
        </div>
        {attributes.gender && attributes.gender !== "unknown" && (
          <Badge variant="secondary" className="capitalize">
            {attributes.gender} · {confidenceLabel(attributes.gender_confidence)}
          </Badge>
        )}
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {groups.map((group) => (
          <div key={group.title} className="space-y-2">
            <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
              {group.title}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {group.items.map((item) => (
                <Badge
                  key={String(item.key)}
                  variant={item.tone === "default" ? "default" : "outline"}
                  className="max-w-full font-normal capitalize"
                >
                  {item.label}: {item.value}
                  {item.confidence !== null && (
                    <span className="ml-1 opacity-70">{confidenceLabel(item.confidence)}</span>
                  )}
                </Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
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

function EvidenceSummary({ tracklets, isLoading }: { tracklets: Tracklet[]; isLoading: boolean }) {
  if (isLoading) {
    return <Skeleton className="h-24 w-full" />;
  }

  if (tracklets.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
        Evidence will appear after matched tracklets are persisted.
      </div>
    );
  }

  const summary = summarizeTracklets(tracklets);

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <SummaryCard label="Avg consistency" value={formatPct(summary.avgConsistency)} />
      <SummaryCard label="Avg good frames" value={formatPct(summary.avgGoodFrameRatio)} />
      <SummaryCard label="Recovery matches" value={summary.recoveryCount.toString()} />
      <SummaryCard label="High-quality tracklets" value={summary.highQualityCount.toString()} />
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-muted/30 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold">{value}</div>
    </div>
  );
}

function SightingsTab({
  personId,
  trackletsById,
}: {
  personId: number;
  trackletsById: Map<string, Tracklet>;
}) {
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
              <TableHead>Evidence</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((s) => {
              const tracklet = trackletsById.get(s.tracklet_id);
              return (
                <TableRow key={s.tracklet_id}>
                  <TableCell>
                    <PersonSnapshot
                      src={s.snapshot_url}
                      alt={`Sighting ${s.tracklet_id} snapshot`}
                      label="Shot"
                      className="h-16 w-12 rounded-md"
                      previewTitle={`Sighting snapshot • ${s.tracklet_id}`}
                      previewDescription={`Device ${s.device_id} • ${formatDateTime(s.started_at)}`}
                    />
                  </TableCell>
                  <TableCell className="text-xs">{s.device_id}</TableCell>
                  <TableCell>{formatDateTime(s.started_at)}</TableCell>
                  <TableCell>{formatDateTime(s.ended_at)}</TableCell>
                  <TableCell className="text-right">{s.duration_seconds.toFixed(1)}</TableCell>
                  <TableCell className="text-right">
                    {(s.quality_score * 100).toFixed(0)}%
                  </TableCell>
                  <TableCell className="min-w-56">
                    {tracklet ? (
                      <div className="space-y-2">
                        <Badge variant="outline">{describeMatchMethod(tracklet.matching)}</Badge>
                        <TrackletEvidenceStrip tracklet={tracklet} compact />
                      </div>
                    ) : (
                      <span className="text-xs text-muted-foreground">
                        Tracklet evidence pending
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function EvidenceTab({ tracklets, isLoading }: { tracklets: Tracklet[]; isLoading: boolean }) {
  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (tracklets.length === 0) {
    return (
      <p className="text-sm text-muted-foreground rounded-lg border border-dashed p-8 text-center">
        No tracklet evidence recorded yet.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {tracklets.map((tracklet) => (
        <Card key={tracklet.tracklet_id}>
          <CardContent className="grid gap-5 p-5 lg:grid-cols-[160px_minmax(0,1fr)]">
            <PersonSnapshot
              src={tracklet.best_crop_url}
              alt={`Tracklet ${tracklet.tracklet_id} best crop`}
              label={`T#${tracklet.track_id}`}
              className="aspect-[4/5]"
              previewTitle={`Tracklet crop • ${tracklet.tracklet_id}`}
              previewDescription={`Best crop from device ${tracklet.device_id}`}
            />
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <div className="text-base font-semibold">
                  {describeMatchMethod(tracklet.matching)}
                </div>
                <Badge
                  variant="outline"
                  className={cn(
                    "text-[11px]",
                    getStatusClasses(
                      tracklet.state === "occlusion_attached" ? "recovering" : "confirmed"
                    )
                  )}
                >
                  {tracklet.state === "occlusion_attached"
                    ? "Occlusion evidence"
                    : getTrackletConfidenceLabel(tracklet.quality)}
                </Badge>
                <Badge variant="secondary">{tracklet.device_id}</Badge>
              </div>

              <p className="text-sm text-muted-foreground">
                {buildLiveEvidenceSummary(
                  tracklet.matching,
                  {
                    v_avg: tracklet.quality.v_avg,
                    embedding_consistency: tracklet.quality.embedding_consistency,
                    overall_consistency: tracklet.quality.overall_consistency,
                    good_frame_ratio: tracklet.quality.good_frame_ratio,
                  },
                  tracklet.quality.v_avg,
                  averageOverlap(tracklet)
                )}
              </p>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <Metric
                  label="Similarity"
                  value={formatDecimal(tracklet.matching.similarity_score)}
                />
                <Metric
                  label="Embedding consistency"
                  value={formatDecimal(tracklet.quality.embedding_consistency)}
                />
                <Metric
                  label="Overall consistency"
                  value={formatPct(tracklet.quality.overall_consistency)}
                />
                <Metric
                  label="Selected / total"
                  value={`${tracklet.evidence.selected_frame_count}/${tracklet.entry_count}`}
                />
              </div>

              <TrackletEvidenceStrip tracklet={tracklet} />

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <Metric label="Average visibility" value={formatDecimal(tracklet.quality.v_avg)} />
                <Metric
                  label="Good frame ratio"
                  value={formatPct(tracklet.quality.good_frame_ratio)}
                />
                <Metric
                  label="Runner-up score"
                  value={formatDecimal(tracklet.matching.runner_up_score)}
                />
                <Metric
                  label="Margin"
                  value={formatDecimal(tracklet.matching.margin_to_runner_up)}
                />
              </div>

              <div className="rounded-lg border bg-muted/20 p-3 text-xs text-muted-foreground">
                <div>
                  Tracklet {tracklet.tracklet_id} spans frames {tracklet.frame_range.start ?? "—"}{" "}
                  to {tracklet.frame_range.end ?? "—"}.
                </div>
                <div>
                  Created {formatDateTime(tracklet.created_at)} with {tracklet.entry_count}{" "}
                  observations and {tracklet.evidence.selected_frame_count} selected frames.
                </div>
                {tracklet.matching.reuse_person_id ? (
                  <div>Recovery hint pointed to person #{tracklet.matching.reuse_person_id}.</div>
                ) : null}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-background px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-sm font-medium">{value}</div>
    </div>
  );
}

function averageOverlap(tracklet: Tracklet): number {
  const samples = tracklet.evidence.frame_samples;
  if (samples.length === 0) return 0;
  return samples.reduce((sum, sample) => sum + sample.overlap_ratio, 0) / samples.length;
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
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {items.map((s) => (
        <Link key={s.person_id} href={`/persons/${s.person_id}`}>
          <Card className="hover:border-primary/40 transition-colors">
            <CardContent className="flex items-center gap-3 p-4">
              <PersonSnapshot
                src={s.person?.snapshot_url}
                alt={`Person ${s.person_id} snapshot`}
                label={`#${s.person_id}`}
                className="h-20 w-16 shrink-0 rounded-md"
                previewTitle={`Similar person #${s.person_id}`}
                previewDescription={`Similarity ${(s.score * 100).toFixed(1)}%`}
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
