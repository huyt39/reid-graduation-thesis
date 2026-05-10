"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { getPersonTimeline } from "@/lib/api";
import type { PaginatedTimeline, TimelineEvent } from "@/types";

function formatDateTime(s: string): string {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
}

function eventColor(type: string): string {
  if (type.includes("start")) return "border-good text-good";
  if (type.includes("end") || type.includes("stop")) return "border-bad text-bad";
  if (type.includes("match") || type.includes("flip")) return "border-accent text-accent";
  return "border-gray-600 text-gray-400";
}

export default function TimelineView({ personId }: { personId: number | null }) {
  const [data, setData] = useState<PaginatedTimeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (personId === null) {
      setData(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPersonTimeline(personId, { page_size: 100 })
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load timeline");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [personId]);

  if (personId === null) {
    return (
      <p className="text-gray-500 text-sm">
        Pick a person above to view their timeline.
      </p>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-gray-500 text-sm">
        <Loader2 size={14} className="animate-spin" /> Loading timeline…
      </div>
    );
  }

  if (error) return <p className="text-bad text-sm">{error}</p>;

  const items: TimelineEvent[] = data?.items ?? [];

  return (
    <div className="flex flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-100 uppercase tracking-wider">
          Timeline · person #{personId}
        </h2>
        {data && <span className="text-xs text-gray-500">{data.total} events</span>}
      </header>

      {items.length === 0 ? (
        <p className="text-gray-500 text-sm">No events recorded.</p>
      ) : (
        <ol className="flex flex-col gap-1.5 border-l border-border pl-4">
          {items.map((e, i) => (
            <li
              key={`${e.timestamp}-${i}`}
              className={`bg-panel border-l-4 ${eventColor(e.event_type)} rounded-r-lg px-3 py-2 flex flex-col gap-0.5`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-100">{e.event_type}</span>
                <span className="text-xs text-gray-400 font-mono">
                  {formatDateTime(e.timestamp)}
                </span>
              </div>
              {e.device_id && (
                <span className="text-xs text-gray-500 font-mono">
                  device {e.device_id}
                </span>
              )}
              {Object.keys(e.details ?? {}).length > 0 && (
                <pre className="text-[11px] text-gray-500 font-mono mt-1 overflow-x-auto">
                  {JSON.stringify(e.details, null, 0)}
                </pre>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
