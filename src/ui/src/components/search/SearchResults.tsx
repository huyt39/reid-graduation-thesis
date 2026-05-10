"use client";

import Link from "next/link";
import type { NLQueryResult } from "@/types";

/**
 * Renders /api/v1/query/natural responses.
 *
 * Backend shape: { parsed_query: {query_type, params}, result: <varies> }.
 * We branch on `data.parsed_query.query_type` to pick the right renderer.
 */
export default function SearchResults({
  data,
  rawError,
}: {
  data: NLQueryResult | null;
  rawError: string | null;
}) {
  if (rawError) {
    return <p className="text-bad text-sm">{rawError}</p>;
  }
  if (!data) return null;

  const qtype = data.parsed_query?.query_type;
  const result = data.result;

  return (
    <div className="flex flex-col gap-3">
      {data.parsed_query && (
        <div className="bg-panel border border-border rounded-lg p-3 text-xs text-gray-400 font-mono">
          <span className="text-gray-500">parsed: </span>
          <span className="text-accent">{qtype}</span>
          <span className="text-gray-600">
            {" "}
            {JSON.stringify(data.parsed_query.params ?? {})}
          </span>
        </div>
      )}

      {qtype === "person_lookup" && <PersonResultBlock result={result} />}
      {qtype === "person_search" && <PersonListResult result={result} />}
      {qtype === "timeline" && <TimelineResultBlock result={result} />}
      {qtype === "similarity_search" && <SimilarityResult result={result} />}
      {qtype === "sighting_aggregation" && <AggregationResult result={result} />}
      {qtype === "device_lookup" && <DeviceListResult result={result} />}
      {(qtype === "error" || qtype === undefined) && (
        <pre className="bg-panel border border-border rounded-lg p-3 text-xs text-gray-400 overflow-auto max-h-96 whitespace-pre-wrap">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

/* ───────── individual result renderers ───────── */

function PersonResultBlock({ result }: { result: unknown }) {
  const p = result as { person_id?: number; attributes?: { gender?: string } } | null;
  if (!p?.person_id) return <EmptyMsg msg="No person found." />;
  return (
    <Link
      href={`/persons/${p.person_id}`}
      className="bg-panel border border-border rounded-lg p-4 hover:border-accent block w-fit"
    >
      <div className="text-sm text-gray-100 font-semibold">#{p.person_id}</div>
      <div className="text-xs text-gray-400 capitalize">
        {p.attributes?.gender ?? "unknown"}
      </div>
    </Link>
  );
}

function PersonListResult({ result }: { result: unknown }) {
  const r = result as { items?: Array<{ person_id: number; attributes?: { gender?: string }; snapshot_url?: string | null }>; total?: number } | null;
  const items = r?.items ?? [];
  if (items.length === 0) return <EmptyMsg msg="No persons matched." />;
  return (
    <div>
      <div className="text-xs text-gray-500 mb-2">{r?.total ?? items.length} matched</div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
        {items.slice(0, 24).map((p) => (
          <Link
            key={p.person_id}
            href={`/persons/${p.person_id}`}
            className="bg-panel border border-border rounded-lg p-2 flex gap-2 hover:border-accent"
          >
            <div className="w-12 h-14 bg-black/40 rounded overflow-hidden shrink-0 flex items-center justify-center">
              {p.snapshot_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={p.snapshot_url} alt="" className="w-full h-full object-cover" />
              ) : (
                <span className="text-gray-600 text-[10px]">—</span>
              )}
            </div>
            <div className="flex flex-col justify-center min-w-0">
              <span className="text-sm font-semibold text-gray-100">#{p.person_id}</span>
              <span className="text-xs text-gray-500 capitalize">
                {p.attributes?.gender ?? "unknown"}
              </span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

function TimelineResultBlock({ result }: { result: unknown }) {
  const r = result as { items?: Array<{ event_type: string; timestamp: string; device_id: string }>; total?: number } | null;
  const items = r?.items ?? [];
  if (items.length === 0) return <EmptyMsg msg="No timeline events." />;
  return (
    <ul className="flex flex-col gap-1.5">
      {items.slice(0, 50).map((e, i) => (
        <li
          key={i}
          className="bg-panel border border-border rounded-lg px-3 py-2 flex justify-between text-sm"
        >
          <span className="text-gray-100">{e.event_type}</span>
          <span className="text-xs text-gray-400 font-mono">
            {e.device_id} · {new Date(e.timestamp).toLocaleString()}
          </span>
        </li>
      ))}
    </ul>
  );
}

function SimilarityResult({ result }: { result: unknown }) {
  const r = result as { similar_persons?: Array<{ person_id: number; score: number; person?: { snapshot_url?: string | null } }> } | null;
  const items = r?.similar_persons ?? [];
  if (items.length === 0) return <EmptyMsg msg="No similar persons above threshold." />;
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
      {items.map((s) => (
        <Link
          key={s.person_id}
          href={`/persons/${s.person_id}`}
          className="bg-panel border border-border rounded-lg p-2 flex gap-2 hover:border-accent"
        >
          <div className="w-12 h-14 bg-black/40 rounded overflow-hidden shrink-0 flex items-center justify-center">
            {s.person?.snapshot_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={s.person.snapshot_url} alt="" className="w-full h-full object-cover" />
            ) : (
              <span className="text-gray-600 text-[10px]">—</span>
            )}
          </div>
          <div className="flex flex-col justify-center">
            <span className="text-sm font-semibold text-gray-100">#{s.person_id}</span>
            <span className="text-xs font-mono text-gray-500">
              {s.score.toFixed(3)}
            </span>
          </div>
        </Link>
      ))}
    </div>
  );
}

function AggregationResult({ result }: { result: unknown }) {
  const r = result as { aggregation?: Array<Record<string, unknown>> } | null;
  const buckets = r?.aggregation ?? [];
  if (buckets.length === 0) return <EmptyMsg msg="Empty aggregation." />;
  const max = Math.max(...buckets.map((b) => Number(b.count ?? 0)), 1);
  return (
    <ul className="flex flex-col gap-1.5">
      {buckets.map((b, i) => {
        const count = Number(b.count ?? 0);
        const labelKey = Object.keys(b).find((k) => k !== "count") ?? "bucket";
        return (
          <li
            key={i}
            className="bg-panel border border-border rounded-lg px-3 py-2 text-sm"
          >
            <div className="flex justify-between items-center mb-1">
              <span className="text-gray-100 font-mono text-xs">
                {String(b[labelKey])}
              </span>
              <span className="text-gray-400 text-xs">{count}</span>
            </div>
            <div className="h-1.5 bg-surface rounded overflow-hidden">
              <div
                className="h-full bg-accent"
                style={{ width: `${(count / max) * 100}%` }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function DeviceListResult({ result }: { result: unknown }) {
  const r = result as { devices?: Array<{ device_id: string; status?: string; sighting_count?: number; last_seen_at?: string | null }> } | null;
  const items = r?.devices ?? [];
  if (items.length === 0) return <EmptyMsg msg="No devices known." />;
  return (
    <ul className="flex flex-col gap-1.5">
      {items.map((d) => (
        <li
          key={d.device_id}
          className="bg-panel border border-border rounded-lg px-3 py-2 flex justify-between text-sm"
        >
          <span className="text-gray-100 font-mono">{d.device_id}</span>
          <span className="text-xs text-gray-400">
            {d.sighting_count ?? 0} sightings · {d.status ?? "?"}
          </span>
        </li>
      ))}
    </ul>
  );
}

function EmptyMsg({ msg }: { msg: string }) {
  return <p className="text-gray-500 text-sm">{msg}</p>;
}
