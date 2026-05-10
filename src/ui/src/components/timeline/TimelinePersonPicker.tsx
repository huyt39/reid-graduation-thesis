"use client";

import { useEffect, useState } from "react";
import { Loader2, Search } from "lucide-react";
import { listPersons } from "@/lib/api";
import type { Person } from "@/types";

interface Props {
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}

export default function TimelinePersonPicker({ selectedId, onSelect }: Props) {
  const [recent, setRecent] = useState<Person[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [idInput, setIdInput] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    listPersons({ page: 1, page_size: 12 })
      .then((res) => {
        if (!cancelled) setRecent(res.items);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load persons");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function submitId(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = idInput.trim();
    if (!trimmed) return;
    const parsed = Number.parseInt(trimmed, 10);
    if (!Number.isNaN(parsed)) onSelect(parsed);
  }

  return (
    <section className="flex flex-col gap-3">
      <form onSubmit={submitId} className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 bg-panel border border-border rounded-lg px-2.5 py-1.5 focus-within:border-accent">
          <Search size={13} className="text-gray-500" />
          <input
            type="text"
            inputMode="numeric"
            placeholder="Person ID…"
            value={idInput}
            onChange={(e) => setIdInput(e.target.value)}
            className="bg-transparent text-sm text-gray-100 placeholder-gray-500 outline-none w-28"
          />
        </div>
        <button
          type="submit"
          className="bg-accent hover:bg-blue-500 text-white rounded-lg px-3 py-1.5 text-sm"
        >
          View timeline
        </button>
        {selectedId !== null && (
          <button
            type="button"
            onClick={() => {
              setIdInput("");
              onSelect(null);
            }}
            className="text-xs text-gray-400 hover:text-gray-100 ml-1"
          >
            Clear
          </button>
        )}
      </form>

      {selectedId === null && (
        <div className="flex flex-col gap-2">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Recent persons
          </h2>

          {loading && (
            <div className="flex items-center gap-2 text-gray-500 text-sm">
              <Loader2 size={14} className="animate-spin" /> Loading…
            </div>
          )}

          {error && <p className="text-bad text-xs">{error}</p>}

          {!loading && !error && recent.length === 0 && (
            <p className="text-gray-500 text-sm">No persons yet.</p>
          )}

          <ul className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
            {recent.map((p) => (
              <li key={p.person_id}>
                <button
                  type="button"
                  onClick={() => onSelect(p.person_id)}
                  className="w-full bg-panel border border-border rounded-lg p-2 flex gap-2 items-center hover:border-accent text-left"
                >
                  <div className="w-10 h-12 bg-black/40 rounded shrink-0 overflow-hidden flex items-center justify-center">
                    {p.snapshot_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={p.snapshot_url}
                        alt={`Person ${p.person_id}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-gray-600 text-[10px]">—</span>
                    )}
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="text-sm text-gray-100 font-semibold">
                      #{p.person_id}
                    </span>
                    <span className="text-xs text-gray-500">
                      {p.stats.sighting_count} sightings
                    </span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
