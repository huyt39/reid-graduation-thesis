"use client";

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, ArrowRight, Expand, ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { TrackletEvidenceStrip } from "@/components/tracklet-evidence-strip";
import { PersonSnapshot } from "@/components/person-snapshot";
import { useOcclusionCandidates } from "@/hooks/use-occlusion-candidates";
import { usePersons } from "@/hooks/use-persons";
import { usePerson, usePersonTracklets } from "@/hooks/use-person";
import {
  buildLiveEvidenceSummary,
  describeMatchMethod,
  formatDecimal,
  formatPct,
} from "@/lib/reid-evidence";
import { cn } from "@/lib/utils";
import type { OcclusionCandidate, Tracklet, TrackletFrameSample } from "@/types";

function pickTracklet(tracklets: Tracklet[]): Tracklet | null {
  const withCrops = tracklets.find((tracklet) =>
    tracklet.evidence.frame_samples.some((sample) => sample.crop_url)
  );
  return withCrops ?? tracklets[0] ?? null;
}

function sampleLabel(sample: TrackletFrameSample): string {
  if (sample.selected) return "Selected for embedding";
  if (sample.selection_reason === "rejected_low_visibility_preview") return "Rejected preview";
  return "Not selected";
}

function EvidenceFrame({
  sample,
  title,
  tone,
  onPreview,
}: {
  sample: TrackletFrameSample | null;
  title: string;
  tone: "selected" | "rejected";
  onPreview?: (sample: TrackletFrameSample) => void;
}) {
  return (
    <div
      className={cn(
        "overflow-hidden rounded-lg border bg-background",
        tone === "selected" ? "border-emerald-400" : "border-rose-300"
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2">
        <div className="text-sm font-medium">{title}</div>
        {sample ? (
          <Badge variant="outline" className="text-[11px]">
            f{sample.frame_idx}
          </Badge>
        ) : null}
      </div>
      {sample?.crop_url ? (
        <button
          type="button"
          onClick={() => onPreview?.(sample)}
          className="group relative block aspect-[4/5] w-full bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
          aria-label={`Preview ${title} frame ${sample.frame_idx}`}
        >
          <Image
            src={sample.crop_url}
            alt={`${title} frame ${sample.frame_idx}`}
            fill
            sizes="(max-width: 768px) 45vw, 320px"
            className="object-cover transition-transform duration-200 group-hover:scale-[1.02]"
          />
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-end bg-gradient-to-t from-black/70 via-black/20 to-transparent px-3 py-2 text-white opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100">
            <Expand className="h-4 w-4" />
          </div>
        </button>
      ) : (
        <div className="relative aspect-[4/5] bg-muted/40">
          <div className="flex h-full items-center justify-center px-4 text-center text-sm text-muted-foreground">
            Crop evidence is not available for this frame yet.
          </div>
        </div>
      )}
      <div className="grid grid-cols-3 gap-2 px-3 py-3 text-xs">
        <Metric label="Vis" value={formatDecimal(sample?.visibility_score)} />
        <Metric label="Overlap" value={formatPct(sample?.overlap_ratio)} />
        <Metric label="Decision" value={sample ? sampleLabel(sample) : "—"} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-xs font-medium">{value}</div>
    </div>
  );
}

function reasonLabel(reason: string): string {
  const labels: Record<string, string> = {
    short_stale_tracklet: "Short occluded track",
    quality_gate_fail: "Quality gate held",
    feature_extraction_failed: "Feature unavailable",
    embedding_consensus_fail: "Embedding conflict",
    quality_gate_blocked_fallback: "Insufficient identity evidence",
    tentative_unconfirmed: "Unconfirmed hypothesis",
  };
  return labels[reason] ?? reason.replaceAll("_", " ");
}

function CandidateCard({ candidate }: { candidate: OcclusionCandidate }) {
  const bestSample =
    candidate.evidence.frame_samples.find((sample) => sample.crop_url) ??
    candidate.evidence.frame_samples[0] ??
    null;
  return (
    <Card className="overflow-hidden">
      <CardContent className="grid grid-cols-[72px_minmax(0,1fr)] gap-3 p-3">
        <PersonSnapshot
          src={candidate.best_crop_url ?? bestSample?.crop_url}
          alt={`Occlusion candidate ${candidate.track_id}`}
          label={`T${candidate.track_id}`}
          className="h-24 w-[72px] rounded-md"
          previewTitle={`Occlusion candidate T${candidate.track_id}`}
          previewDescription="Unconfirmed evidence is intentionally separated from confirmed persons."
        />
        <div className="min-w-0 space-y-2">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="text-sm font-medium">Track #{candidate.track_id}</div>
              <div className="text-xs text-muted-foreground">
                frames {candidate.frame_range.start ?? "?"}-{candidate.frame_range.end ?? "?"}
              </div>
            </div>
            <Badge variant="outline" className="border-amber-300 bg-amber-50 text-amber-700">
              {candidate.status}
            </Badge>
          </div>
          <p className="text-xs leading-4 text-muted-foreground">
            {reasonLabel(candidate.reason)}. Kept out of confirmed persons until stronger evidence
            appears.
          </p>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <Metric label="Frames" value={candidate.entry_count.toString()} />
            <Metric label="Vis" value={formatDecimal(candidate.quality.v_avg)} />
            <Metric label="Emb" value={formatDecimal(candidate.quality.embedding_consistency)} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function OcclusionDemo() {
  const candidatePageSize = 24;
  const {
    data: personsData,
    isLoading: isPersonsLoading,
    error: personsError,
  } = usePersons({
    page: 1,
    page_size: 12,
  });
  const [candidatePage, setCandidatePage] = useState(1);
  const {
    data: candidatesData,
    isLoading: isCandidatesLoading,
    error: candidatesError,
  } = useOcclusionCandidates({
    status: "unconfirmed",
    page: candidatePage,
    page_size: candidatePageSize,
  });
  const persons = useMemo(() => personsData?.items ?? [], [personsData]);
  const candidates = candidatesData?.items ?? [];
  const candidateTotal = candidatesData?.total ?? 0;
  const candidateTotalPages = Math.max(1, Math.ceil(candidateTotal / candidatePageSize));
  const [selectedPersonId, setSelectedPersonId] = useState<number | null>(null);
  const [previewSample, setPreviewSample] = useState<TrackletFrameSample | null>(null);

  useEffect(() => {
    if (selectedPersonId === null && persons.length > 0) {
      setSelectedPersonId(persons[0].person_id);
    }
  }, [persons, selectedPersonId]);

  const { data: person } = usePerson(selectedPersonId);
  const { data: trackletsData, isLoading: isTrackletsLoading } =
    usePersonTracklets(selectedPersonId);
  const tracklet = useMemo(() => pickTracklet(trackletsData?.items ?? []), [trackletsData]);

  const rejectedSample =
    tracklet?.evidence.frame_samples.find(
      (sample) => !sample.selected && sample.selection_reason === "rejected_low_visibility_preview"
    ) ??
    tracklet?.evidence.frame_samples.find((sample) => !sample.selected) ??
    null;
  const selectedSample =
    tracklet?.evidence.frame_samples.find((sample) => sample.selected && sample.crop_url) ??
    tracklet?.evidence.frame_samples.find((sample) => sample.selected) ??
    null;

  if (isPersonsLoading) {
    return <Skeleton className="h-[560px] w-full" />;
  }

  if (personsError) {
    return <p className="text-sm text-destructive">{personsError.message}</p>;
  }

  if (persons.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-10 text-center text-sm text-muted-foreground">
        No persons are available for occlusion evidence yet.
      </div>
    );
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[280px_minmax(0,1fr)]">
      <aside className="space-y-3">
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">
              Occlusion candidates
            </div>
            <Badge variant="secondary">{candidateTotal}</Badge>
          </div>
          {isCandidatesLoading ? (
            <Skeleton className="h-28 w-full" />
          ) : candidatesError ? (
            <p className="text-xs text-destructive">{candidatesError.message}</p>
          ) : candidates.length === 0 ? (
            <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground">
              No unresolved occlusion candidates yet.
            </div>
          ) : (
            <div className="space-y-3">
              <div className="grid max-h-[calc(100vh-280px)] gap-2 overflow-y-auto pr-1">
                {candidates.map((candidate) => (
                  <CandidateCard key={candidate.candidate_id} candidate={candidate} />
                ))}
              </div>
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs text-muted-foreground">
                  Page {candidatePage}/{candidateTotalPages}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setCandidatePage((page) => Math.max(1, page - 1))}
                    disabled={candidatePage <= 1}
                  >
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setCandidatePage((page) => Math.min(candidateTotalPages, page + 1))
                    }
                    disabled={candidatePage >= candidateTotalPages}
                  >
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
      </aside>

      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Recent persons</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
              {persons.map((item) => (
                <button
                  key={item.person_id}
                  type="button"
                  onClick={() => setSelectedPersonId(item.person_id)}
                  className={cn(
                    "rounded-lg border bg-background px-3 py-2 text-left text-sm transition-colors",
                    selectedPersonId === item.person_id
                      ? "border-primary bg-primary/5"
                      : "hover:border-primary/40"
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium">Person #{item.person_id}</span>
                    <Badge variant={item.is_active ? "default" : "secondary"}>
                      {item.is_active ? "Active" : "Inactive"}
                    </Badge>
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {item.stats.sighting_count.toLocaleString()} sightings •{" "}
                    {item.stats.last_seen_device || "—"}
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base">
                {person ? `Person #${person.person_id}` : "Occlusion case"}
              </CardTitle>
              <p className="mt-1 text-sm text-muted-foreground">
                {tracklet
                  ? describeMatchMethod(tracklet.matching)
                  : "Choose a person with persisted tracklet evidence."}
              </p>
            </div>
            {selectedPersonId ? (
              <Button asChild variant="outline" size="sm">
                <Link href={`/persons/${selectedPersonId}`}>
                  <ExternalLink className="h-4 w-4" />
                  Person detail
                </Link>
              </Button>
            ) : null}
          </CardHeader>
          <CardContent>
            {isTrackletsLoading ? (
              <Skeleton className="h-[420px] w-full" />
            ) : tracklet ? (
              <div className="space-y-5">
                <div className="grid items-start gap-4 lg:grid-cols-[1fr_auto_1fr]">
                  <EvidenceFrame
                    sample={rejectedSample}
                    title="Rejected / occluded"
                    tone="rejected"
                    onPreview={setPreviewSample}
                  />
                  <div className="hidden h-full items-center lg:flex">
                    <ArrowRight className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <EvidenceFrame
                    sample={selectedSample}
                    title="Selected / identity evidence"
                    tone="selected"
                    onPreview={setPreviewSample}
                  />
                </div>

                <div className="rounded-lg border bg-muted/20 p-3 text-sm text-muted-foreground">
                  {buildLiveEvidenceSummary(
                    tracklet.matching,
                    {
                      v_avg: tracklet.quality.v_avg,
                      embedding_consistency: tracklet.quality.embedding_consistency,
                      overall_consistency: tracklet.quality.overall_consistency,
                      good_frame_ratio: tracklet.quality.good_frame_ratio,
                    },
                    tracklet.quality.v_avg,
                    rejectedSample?.overlap_ratio ?? 0
                  )}
                </div>

                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <Metric
                    label="Similarity"
                    value={formatDecimal(tracklet.matching.similarity_score)}
                  />
                  <Metric
                    label="Good frame ratio"
                    value={formatPct(tracklet.quality.good_frame_ratio)}
                  />
                  <Metric
                    label="Embedding consistency"
                    value={formatDecimal(tracklet.quality.embedding_consistency)}
                  />
                  <Metric
                    label="Selected / total"
                    value={`${tracklet.evidence.selected_frame_count}/${tracklet.entry_count}`}
                  />
                </div>

                <TrackletEvidenceStrip tracklet={tracklet} />
              </div>
            ) : (
              <div className="rounded-lg border border-dashed p-10 text-center text-sm text-muted-foreground">
                This person does not have persisted tracklet evidence yet.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Dialog
        open={previewSample !== null}
        onOpenChange={(open) => !open && setPreviewSample(null)}
      >
        <DialogContent className="max-h-[92vh] max-w-4xl overflow-hidden p-0 sm:max-w-5xl">
          <DialogHeader className="px-6 pt-6 pb-0">
            <DialogTitle>
              Frame {previewSample?.frame_idx} —{" "}
              {previewSample ? sampleLabel(previewSample) : ""}
            </DialogTitle>
            <DialogDescription>
              {previewSample
                ? `Visibility ${formatDecimal(previewSample.visibility_score)} · Overlap ${formatPct(previewSample.overlap_ratio)}`
                : ""}
            </DialogDescription>
          </DialogHeader>
          <div className="px-6 pb-6 pt-4">
            <div className="relative overflow-hidden rounded-lg border bg-muted/30">
              <div className="relative aspect-[4/5] max-h-[72vh] min-h-80 w-full">
                {previewSample?.crop_url ? (
                  <Image
                    src={previewSample.crop_url}
                    alt={`Frame ${previewSample.frame_idx}`}
                    fill
                    sizes="100vw"
                    className="object-contain"
                  />
                ) : null}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
