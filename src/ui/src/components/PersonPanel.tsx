"use client";

import { Users } from "lucide-react";
import type { TrackedPerson } from "@/hooks/useWebSocket";

const LIVE_ATTRIBUTE_THRESHOLD = 0.9;

function visColor(score: number): string {
  if (score >= 0.7) return "text-good";
  if (score >= 0.4) return "text-mid";
  return "text-bad";
}

function stateLabel(state: string | null): string {
  if (!state) return "";
  if (state === "confirmed") return "✓";
  if (state === "tentative") return "~";
  return state.slice(0, 1).toUpperCase();
}

function collectAttributeBadges(person: TrackedPerson): string[] {
  const badges: string[] = [];
  const maybeAdd = (label: string, value: string | undefined, confidence: number | undefined) => {
    if (!value || value === "unknown" || (confidence ?? 0) < LIVE_ATTRIBUTE_THRESHOLD) return;
    badges.push(`${label}: ${value}`);
  };

  maybeAdd("age", person.age_child, person.age_child_confidence);
  maybeAdd("backpack", person.backpack, person.backpack_confidence);
  maybeAdd("sidebag", person.sidebag, person.sidebag_confidence);
  maybeAdd("hat", person.hat, person.hat_confidence);
  maybeAdd("glasses", person.glasses, person.glasses_confidence);
  maybeAdd("sleeve", person.sleeve, person.sleeve_confidence);
  maybeAdd("lower", person.lower, person.lower_confidence);

  return badges;
}

interface Props {
  persons: TrackedPerson[];
}

export default function PersonPanel({ persons }: Props) {
  return (
    <aside className="flex flex-col gap-2 w-56 shrink-0">
      <div className="flex items-center gap-2 text-xs text-gray-400 uppercase tracking-wider px-1">
        <Users size={13} />
        Active Persons ({persons.length})
      </div>

      <div className="flex flex-col gap-1.5 overflow-y-auto">
        {persons.length === 0 && (
          <p className="text-xs text-gray-500 px-1">No detections</p>
        )}
        {persons.map((p) => {
          const isTentative = p.tracklet_state === "tentative";
          return (
          <div
            key={`${p.person_id}-${p.tracklet_id ?? ""}`}
            className={`bg-panel border rounded-lg px-3 py-2 flex flex-col gap-1 ${isTentative ? "border-slate-600 opacity-60" : "border-border"}`}
          >
            <div className="flex items-center justify-between">
              <span className={`font-semibold text-sm ${isTentative ? "text-slate-400" : "text-gray-100"}`}>
                {isTentative ? `?` : `#${p.person_id}`}
                {p.tracklet_state && !isTentative && (
                  <span className="ml-1 text-xs text-gray-500">
                    {stateLabel(p.tracklet_state)}
                  </span>
                )}
              </span>
              <span className="text-xs text-gray-500 font-mono">
                {(p.confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-500">vis</span>
              <span className={`font-mono ${visColor(p.visibility_score)}`}>
                {p.visibility_score.toFixed(2)}
              </span>
            </div>

            {p.quality && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-500">consist</span>
                <span className="font-mono text-gray-400">
                  {p.quality.embedding_consistency.toFixed(2)}
                </span>
              </div>
            )}

            {collectAttributeBadges(p).length > 0 && (
              <div className="flex flex-wrap gap-1 pt-0.5">
                {collectAttributeBadges(p).map((badge) => (
                  <span
                    key={badge}
                    className="rounded-full border border-border bg-black/30 px-1.5 py-0.5 text-[10px] text-gray-300"
                  >
                    {badge}
                  </span>
                ))}
              </div>
            )}
          </div>
        );})}
      </div>
    </aside>
  );
}
