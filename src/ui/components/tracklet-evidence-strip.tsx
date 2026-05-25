"use client";

import { useState } from "react";
import Image from "next/image";
import { Expand } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { Tracklet, TrackletFrameSample } from "@/types";

function reasonLabel(reason: string, selected: boolean): string {
  if (selected) return "Selected";
  if (reason === "rejected_low_visibility_preview") return "Rejected";
  return "Skipped";
}

function scoreColor(score: number): string {
  if (score >= 0.7) return "bg-emerald-500/80";
  if (score >= 0.45) return "bg-amber-400/80";
  return "bg-rose-400/80";
}

export function TrackletEvidenceStrip({
  tracklet,
  compact = false,
}: {
  tracklet: Tracklet;
  compact?: boolean;
}) {
  const samples = tracklet.evidence.frame_samples;
  const [previewSample, setPreviewSample] = useState<TrackletFrameSample | null>(null);

  if (samples.length === 0) {
    return <div className="text-xs text-muted-foreground">No frame-level evidence saved.</div>;
  }

  const hasCrops = samples.some((sample) => sample.crop_url);

  if (hasCrops && !compact) {
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
          {samples
            .filter((sample) => sample.crop_url || sample.selected)
            .slice(0, 12)
            .map((sample) => (
              <div
                key={sample.frame_idx}
                className={cn(
                  "overflow-hidden rounded-md border bg-background",
                  sample.selected ? "border-emerald-500" : "border-rose-300"
                )}
              >
                {sample.crop_url ? (
                  <button
                    type="button"
                    onClick={() => setPreviewSample(sample)}
                    className="group relative block aspect-[4/5] w-full bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1"
                    aria-label={`Preview frame ${sample.frame_idx}`}
                  >
                    <Image
                      src={sample.crop_url}
                      alt={`Frame ${sample.frame_idx}`}
                      fill
                      sizes="160px"
                      className="object-cover transition-transform duration-200 group-hover:scale-[1.02]"
                    />
                    <div className="absolute inset-x-0 bottom-0 flex items-center justify-end bg-gradient-to-t from-black/70 via-black/20 to-transparent px-2 py-1.5 text-white opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100">
                      <Expand className="h-3.5 w-3.5" />
                    </div>
                  </button>
                ) : (
                  <div className="relative aspect-[4/5] bg-muted/40">
                    <div className="flex h-full items-center justify-center text-[11px] text-muted-foreground">
                      No crop
                    </div>
                  </div>
                )}
                <div className="space-y-1 px-2 py-1.5 text-[11px]">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium">f{sample.frame_idx}</span>
                    <span className={sample.selected ? "text-emerald-700" : "text-rose-700"}>
                      {reasonLabel(sample.selection_reason, sample.selected)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-muted-foreground">
                    <span>vis {sample.visibility_score.toFixed(2)}</span>
                    <span>ovl {(sample.overlap_ratio * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            ))}
        </div>
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span>Rejected crops show occluded evidence</span>
          <span>Selected crops form the identity embedding</span>
        </div>

        <Dialog open={previewSample !== null} onOpenChange={(open) => !open && setPreviewSample(null)}>
          <DialogContent className="max-h-[92vh] max-w-4xl overflow-hidden p-0 sm:max-w-5xl">
            <DialogHeader className="px-6 pt-6 pb-0">
              <DialogTitle>
                Frame {previewSample?.frame_idx} —{" "}
                {previewSample
                  ? reasonLabel(previewSample.selection_reason, previewSample.selected)
                  : ""}
              </DialogTitle>
              <DialogDescription>
                {previewSample
                  ? `Visibility ${previewSample.visibility_score.toFixed(2)} · Overlap ${(previewSample.overlap_ratio * 100).toFixed(0)}%`
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

  return (
    <div className="space-y-2">
      <div className={cn("flex items-end gap-1", compact ? "h-8" : "h-14")}>
        {samples.map((sample) => {
          const heightClass = compact ? "min-h-3" : "min-h-4";
          return (
            <div key={sample.frame_idx} className="flex min-w-0 flex-1 flex-col items-center gap-1">
              <div
                className={cn(
                  "w-full rounded-sm border transition-colors",
                  heightClass,
                  sample.selected ? "border-slate-900" : "border-transparent",
                  scoreColor(sample.visibility_score)
                )}
                style={{ height: `${Math.max(20, sample.visibility_score * 100)}%` }}
                title={`frame ${sample.frame_idx} - vis ${sample.visibility_score.toFixed(2)} - overlap ${sample.overlap_ratio.toFixed(2)}${sample.selected ? " - selected" : ""}`}
              />
            </div>
          );
        })}
      </div>
      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>Occluded / weak</span>
        <span>Selected frames are outlined</span>
        <span>Clean / strong</span>
      </div>
    </div>
  );
}
