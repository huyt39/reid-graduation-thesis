"use client";

import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { getPersonSightings } from "@/lib/api";
import type { PaginatedSightings } from "@/types";

const PAGE_SIZE = 10;

function formatDateTime(s: string): string {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

export default function SightingHistory({ personId }: { personId: number }) {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<PaginatedSightings | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPersonSightings(personId, page, PAGE_SIZE)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load sightings");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [personId, page]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <section className="flex flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-100 uppercase tracking-wider">
          Sightings
        </h2>
        {data && (
          <span className="text-xs text-gray-500">{data.total} total</span>
        )}
      </header>

      {loading && !data && (
        <div className="flex items-center gap-2 text-gray-500 text-sm">
          <Loader2 size={14} className="animate-spin" /> Loading sightings…
        </div>
      )}

      {error && <p className="text-bad text-xs">{error}</p>}

      <ul className="flex flex-col gap-2">
        {(data?.items ?? []).map((s) => (
          <li
            key={s.tracklet_id}
            className="bg-panel border border-border rounded-lg p-3 flex gap-3"
          >
            <div className="w-16 h-20 bg-black/40 rounded overflow-hidden shrink-0 flex items-center justify-center">
              {s.snapshot_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={s.snapshot_url}
                  alt={`tracklet ${s.tracklet_id.slice(0, 6)}`}
                  className="w-full h-full object-cover"
                />
              ) : (
                <span className="text-gray-600 text-[10px]">no img</span>
              )}
            </div>
            <div className="flex-1 flex flex-col gap-0.5 min-w-0">
              <div className="flex items-center justify-between text-sm text-gray-100">
                <span className="font-mono">{s.device_id}</span>
                <span className="text-xs text-gray-500">
                  {formatDuration(s.duration_seconds)}
                </span>
              </div>
              <div className="text-xs text-gray-400">
                {formatDateTime(s.started_at)}
              </div>
              <div className="flex items-center gap-3 text-xs text-gray-500">
                <span>quality {(s.quality_score * 100).toFixed(0)}%</span>
                <span className="font-mono truncate" title={s.tracklet_id}>
                  {s.tracklet_id.slice(0, 8)}…
                </span>
              </div>
            </div>
          </li>
        ))}
        {!loading && data && data.items.length === 0 && (
          <li className="text-gray-500 text-sm">No sightings recorded.</li>
        )}
      </ul>

      <div className="flex items-center justify-center gap-3 text-sm text-gray-300">
        <button
          type="button"
          disabled={page <= 1 || loading}
          onClick={() => setPage(page - 1)}
          className="flex items-center gap-1 px-2 py-1 rounded disabled:opacity-30 hover:bg-panel"
        >
          <ChevronLeft size={14} /> Prev
        </button>
        <span className="text-gray-500">
          page {page} of {totalPages}
        </span>
        <button
          type="button"
          disabled={page >= totalPages || loading}
          onClick={() => setPage(page + 1)}
          className="flex items-center gap-1 px-2 py-1 rounded disabled:opacity-30 hover:bg-panel"
        >
          Next <ChevronRight size={14} />
        </button>
      </div>
    </section>
  );
}
