"use client";

import { Users } from "lucide-react";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PersonSnapshot } from "@/components/person-snapshot";
import { cn } from "@/lib/utils";
import {
  buildLiveEvidenceSummary,
  formatDecimal,
  formatPct,
  getLiveStatusLabel,
  getStatusClasses,
} from "@/lib/reid-evidence";
import type { TrackedPerson } from "@/hooks/use-websocket";

// Default threshold for most attributes. Glasses uses a lower value because
// the PA-100K model's glasses head outputs calibrated confidence in the 0.73–0.84
// range — systematically below the default — while still being accurate.
const LIVE_ATTRIBUTE_THRESHOLD = 0.9;
const GLASSES_THRESHOLD = 0.72;

// Labels that represent the *absence* of an attribute — not useful to display.
const NEGATIVE_LABELS = new Set(["no_glasses", "no_backpack", "no_sidebag", "no_hat", "adult"]);

function visClass(score: number): string {
  if (score >= 0.7) return "text-emerald-600";
  if (score >= 0.4) return "text-amber-600";
  return "text-destructive";
}

function stateLabel(state: string | null): string {
  if (!state) return "";
  if (state === "confirmed") return "✓";
  if (state === "tentative") return "~";
  if (state === "raw_edge") return "R";
  return state.slice(0, 1).toUpperCase();
}

function getPersonDisplayLabel(person: TrackedPerson): string {
  if (person.tracklet_state === "tentative") {
    return person.track_id != null ? `T#${person.track_id}` : "?";
  }
  if (person.person_id === null) return "Raw";
  return `#${person.person_id}`;
}

function collectAttributeBadges(person: TrackedPerson): string[] {
  const badges: string[] = [];
  const maybeAdd = (
    label: string,
    value: string | undefined,
    confidence: number | undefined,
    threshold = LIVE_ATTRIBUTE_THRESHOLD
  ) => {
    if (!value || value === "unknown" || NEGATIVE_LABELS.has(value)) return;
    if ((confidence ?? 0) < threshold) return;
    badges.push(`${label}: ${value}`);
  };

  maybeAdd("age", person.age_child, person.age_child_confidence);
  maybeAdd("backpack", person.backpack, person.backpack_confidence);
  maybeAdd("sidebag", person.sidebag, person.sidebag_confidence);
  maybeAdd("hat", person.hat, person.hat_confidence);
  // Glasses temporarily hidden — the PA-100K head's positive class is
  // unreliable in surveillance footage and no post-processing approach
  // has improved it. Data still flows from worker → Kafka → DB; this
  // line is the single switch to re-enable display once the underlying
  // model is finetuned or replaced.
  // maybeAdd("glasses", person.glasses, person.glasses_confidence, GLASSES_THRESHOLD);
  maybeAdd("sleeve", person.sleeve, person.sleeve_confidence);
  maybeAdd("lower", person.lower, person.lower_confidence);

  return badges;
}

export function PersonsPanel({ persons }: { persons: TrackedPerson[] }) {
  return (
    <aside className="w-full lg:w-72 shrink-0 flex flex-col gap-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground uppercase tracking-wider px-1">
        <Users className="h-3.5 w-3.5" />
        Active persons ({persons.length})
      </div>

      <div className="flex flex-col gap-2 overflow-y-auto">
        {persons.length === 0 && (
          <p className="text-xs text-muted-foreground px-1">No detections</p>
        )}
        {persons.map((p) => {
          const isTentative = p.tracklet_state === "tentative";
          const isRaw = p.person_id === null;
          const badges = collectAttributeBadges(p);
          const status =
            p.status ?? (isTentative ? "tentative" : isRaw ? "recovering" : "confirmed");
          const displayLabel = getPersonDisplayLabel(p);
          return (
            <Card
              key={`${p.live_track_key ?? p.tracklet_id ?? p.person_id ?? "raw"}-${p.bbox.join("-")}`}
              className={cn("py-3 gap-2", isTentative && "opacity-60")}
            >
              <CardContent className="grid grid-cols-[52px_minmax(0,1fr)] gap-3 px-3">
                <PersonSnapshot
                  src={p.snapshot_url}
                  alt={`Tracked person ${p.person_id ?? "raw"}`}
                  label={displayLabel}
                  className="h-[72px] w-[52px] rounded-md"
                  previewTitle={
                    isRaw ? "Raw detection crop unavailable" : `Tracked person ${displayLabel}`
                  }
                />
                <div className="space-y-1 text-xs">
                  <CardTitle className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <span className={cn("font-semibold", isTentative && "text-muted-foreground")}>
                        {displayLabel}
                        {p.tracklet_state && !isTentative && (
                          <span className="ml-1 text-xs text-muted-foreground">
                            {stateLabel(p.tracklet_state)}
                          </span>
                        )}
                      </span>
                      <Badge
                        variant="outline"
                        className={cn("text-[10px]", getStatusClasses(status))}
                      >
                        {getLiveStatusLabel(status)}
                      </Badge>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {(p.confidence * 100).toFixed(0)}%
                    </span>
                  </CardTitle>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">live vis</span>
                    <span className={visClass(p.live_visibility_score)}>
                      {formatDecimal(p.live_visibility_score)}
                    </span>
                  </div>
                  {p.quality && (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">tracklet vis</span>
                        <span className="text-muted-foreground">
                          {formatDecimal(p.quality.v_avg)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">consist</span>
                        <span className="text-muted-foreground">
                          {formatDecimal(p.quality.embedding_consistency)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">good frames</span>
                        <span className="text-muted-foreground">
                          {formatPct(p.quality.good_frame_ratio)}
                        </span>
                      </div>
                    </>
                  )}
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">overlap</span>
                    <span className={visClass(1 - p.overlap_ratio)}>
                      {formatPct(p.overlap_ratio)}
                    </span>
                  </div>
                  <p className="pt-1 text-[11px] leading-4 text-muted-foreground">
                    {isRaw
                      ? "Raw edge detection. ReID evidence and identity assignment are not available yet."
                      : buildLiveEvidenceSummary(
                          p.matching,
                          p.quality,
                          p.live_visibility_score,
                          p.overlap_ratio
                        )}
                  </p>
                  {badges.length > 0 && (
                    <div className="flex flex-wrap gap-1 pt-1">
                      {badges.map((badge) => (
                        <Badge key={badge} variant="outline" className="text-[10px] font-normal">
                          {badge}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </aside>
  );
}
