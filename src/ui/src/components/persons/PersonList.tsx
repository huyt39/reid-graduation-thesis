"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ChevronLeft, ChevronRight, Loader2, RefreshCw } from "lucide-react";
import { listPersons } from "@/lib/api";
import type { PaginatedPersons, Person } from "@/types";

const PAGE_SIZE = 20;

export default function PersonList() {
  const [filters, setFilters] = useState<{
    gender: string;
    device: string;
    is_active: string;
  }>({ gender: "", device: "", is_active: "" });
  const [page, setPage] = useState(1);
  const [data, setData] = useState<PaginatedPersons | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const opts: Parameters<typeof listPersons>[0] = {
        page,
        page_size: PAGE_SIZE,
      };
      if (filters.gender) opts.gender = filters.gender;
      if (filters.device) opts.device = filters.device;
      if (filters.is_active) opts.is_active = filters.is_active === "true";
      const res = await listPersons(opts);
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load persons");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div className="flex flex-col gap-4 p-5">
      {/* Filter row */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (page === 1) load();
          else setPage(1);
        }}
        className="flex flex-wrap items-center gap-2"
      >
        <select
          value={filters.gender}
          onChange={(e) => setFilters({ ...filters, gender: e.target.value })}
          className="bg-panel border border-border rounded-lg px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-accent"
        >
          <option value="">Any gender</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
          <option value="unknown">Unknown</option>
        </select>

        <input
          type="text"
          placeholder="Last-seen device"
          value={filters.device}
          onChange={(e) => setFilters({ ...filters, device: e.target.value })}
          className="bg-panel border border-border rounded-lg px-3 py-1.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-accent w-44"
        />

        <select
          value={filters.is_active}
          onChange={(e) => setFilters({ ...filters, is_active: e.target.value })}
          className="bg-panel border border-border rounded-lg px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:border-accent"
        >
          <option value="">Any status</option>
          <option value="true">Active only</option>
          <option value="false">Inactive only</option>
        </select>

        <button
          type="submit"
          className="bg-accent hover:bg-blue-500 text-white rounded-lg px-3 py-1.5 text-sm flex items-center gap-1.5"
          disabled={loading}
        >
          {loading ? (
            <Loader2 size={14} className="animate-spin" />
          ) : (
            <RefreshCw size={14} />
          )}
          Apply
        </button>

        {data && (
          <span className="text-xs text-gray-500 ml-auto">
            {data.total} matched
          </span>
        )}
      </form>

      {error && <p className="text-bad text-xs">{error}</p>}

      {/* Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3">
        {(data?.items ?? []).map((p) => (
          <PersonCard key={p.person_id} person={p} />
        ))}
        {!loading && data && data.items.length === 0 && (
          <p className="text-gray-500 text-sm col-span-full">
            No persons match the current filters.
          </p>
        )}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-center gap-3 text-sm text-gray-300 mt-2">
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
    </div>
  );
}

function PersonCard({ person }: { person: Person }) {
  const { gender, gender_confidence } = person.attributes;
  return (
    <Link
      href={`/persons/${person.person_id}`}
      className="bg-panel border border-border rounded-xl overflow-hidden hover:border-accent transition-colors flex flex-col"
    >
      <div className="aspect-[3/4] bg-black/40 flex items-center justify-center">
        {person.snapshot_url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={person.snapshot_url}
            alt={`Person ${person.person_id}`}
            className="w-full h-full object-cover"
          />
        ) : (
          <span className="text-gray-600 text-xs">no snapshot</span>
        )}
      </div>
      <div className="px-3 py-2 flex flex-col gap-0.5">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-sm text-gray-100">
            #{person.person_id}
          </span>
          <span className={person.is_active ? "text-good text-xs" : "text-gray-500 text-xs"}>
            {person.is_active ? "active" : "inactive"}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs text-gray-400">
          <span className="capitalize">{gender || "unknown"}</span>
          {gender_confidence > 0 && (
            <span className="font-mono">
              {(gender_confidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <div className="text-xs text-gray-500">
          {person.stats.sighting_count} sightings
          {person.stats.last_seen_device && (
            <span> · {person.stats.last_seen_device}</span>
          )}
        </div>
      </div>
    </Link>
  );
}
