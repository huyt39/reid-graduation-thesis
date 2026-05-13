"use client";

import { Users } from "lucide-react";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PersonSnapshot } from "@/components/person-snapshot";
import { cn } from "@/lib/utils";
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
  return state.slice(0, 1).toUpperCase();
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
  maybeAdd("glasses", person.glasses, person.glasses_confidence, GLASSES_THRESHOLD);
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
          const badges = collectAttributeBadges(p);
          return (
            <Card
              key={`${p.person_id}-${p.tracklet_id ?? ""}`}
              className={cn("py-3 gap-2", isTentative && "opacity-60")}
            >
              <CardContent className="grid grid-cols-[52px_minmax(0,1fr)] gap-3 px-3">
                <PersonSnapshot
                  src={p.snapshot_url}
                  alt={`Tracked person ${p.person_id}`}
                  label={isTentative ? "?" : `#${p.person_id}`}
                  className="h-[72px] w-[52px] rounded-md"
                />
                <div className="space-y-1 text-xs">
                  <CardTitle className="flex items-center justify-between text-sm">
                    <span className={cn("font-semibold", isTentative && "text-muted-foreground")}>
                      {isTentative ? "?" : `#${p.person_id}`}
                      {p.tracklet_state && !isTentative && (
                        <span className="ml-1 text-xs text-muted-foreground">
                          {stateLabel(p.tracklet_state)}
                        </span>
                      )}
                    </span>
                    <span className="font-mono text-xs text-muted-foreground">
                      {(p.confidence * 100).toFixed(0)}%
                    </span>
                  </CardTitle>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">vis</span>
                    <span className={cn("font-mono", visClass(p.visibility_score))}>
                      {p.visibility_score.toFixed(2)}
                    </span>
                  </div>
                  {p.quality && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">consist</span>
                      <span className="font-mono text-muted-foreground">
                        {p.quality.embedding_consistency.toFixed(2)}
                      </span>
                    </div>
                  )}
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
