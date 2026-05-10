"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Clock, Loader2, Users } from "lucide-react";
import { getPerson, getSimilarPersons } from "@/lib/api";
import type { Person, SimilarPersonItem } from "@/types";
import SightingHistory from "./SightingHistory";

function formatDateTime(s: string | null): string {
  if (!s) return "—";
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
}

export default function PersonDetail({ personId }: { personId: number }) {
  const [person, setPerson] = useState<Person | null>(null);
  const [similar, setSimilar] = useState<SimilarPersonItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([
      getPerson(personId),
      getSimilarPersons(personId, 6).catch(() => ({ similar_persons: [] })),
    ])
      .then(([p, sim]) => {
        if (!cancelled) {
          setPerson(p);
          setSimilar(sim.similar_persons);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load person");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [personId]);

  if (loading) {
    return (
      <div className="p-5 flex items-center gap-2 text-gray-500 text-sm">
        <Loader2 size={14} className="animate-spin" /> Loading person…
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-5">
        <Link
          href="/persons"
          className="text-xs text-gray-400 hover:text-gray-100 flex items-center gap-1 mb-3"
        >
          <ArrowLeft size={13} /> Back
        </Link>
        <p className="text-bad text-sm">{error}</p>
      </div>
    );
  }

  if (!person) return null;

  const { attributes, stats } = person;

  return (
    <div className="p-5 flex flex-col gap-5 max-w-5xl">
      <Link
        href="/persons"
        className="text-xs text-gray-400 hover:text-gray-100 flex items-center gap-1"
      >
        <ArrowLeft size={13} /> Back to persons
      </Link>

      {/* Header */}
      <header className="flex gap-4">
        <div className="w-32 h-40 bg-black/40 rounded-xl overflow-hidden shrink-0 flex items-center justify-center border border-border">
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

        <div className="flex flex-col gap-1.5 flex-1 min-w-0">
          <h1 className="text-2xl font-bold text-gray-100">
            Person #{person.person_id}
          </h1>
          <div className="flex flex-wrap items-center gap-3 text-sm text-gray-400">
            <span className="capitalize">
              {attributes.gender || "unknown"}
              {attributes.gender_confidence > 0 && (
                <span className="ml-1 text-gray-500 text-xs font-mono">
                  {(attributes.gender_confidence * 100).toFixed(0)}%
                </span>
              )}
            </span>
            <span>·</span>
            <span className={person.is_active ? "text-good" : "text-gray-500"}>
              {person.is_active ? "active" : "inactive"}
            </span>
            <span>·</span>
            <span>{stats.sighting_count} sightings</span>
          </div>
          <div className="text-xs text-gray-500 flex flex-wrap gap-x-4 gap-y-1 mt-2">
            <span>
              <span className="text-gray-600">first seen:</span>{" "}
              {formatDateTime(stats.first_seen_at)}
            </span>
            <span>
              <span className="text-gray-600">last seen:</span>{" "}
              {formatDateTime(stats.last_seen_at)}
            </span>
            {stats.last_seen_device && (
              <span>
                <span className="text-gray-600">last device:</span>{" "}
                <span className="font-mono">{stats.last_seen_device}</span>
              </span>
            )}
          </div>

          <div className="flex gap-2 mt-3">
            <Link
              href={`/timeline?person_id=${person.person_id}`}
              className="bg-panel border border-border hover:border-accent rounded-lg px-3 py-1.5 text-xs text-gray-200 flex items-center gap-1.5"
            >
              <Clock size={13} /> Timeline
            </Link>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2">
          <SightingHistory personId={person.person_id} />
        </div>

        <aside className="flex flex-col gap-3">
          <section className="bg-panel border border-border rounded-xl p-3 flex flex-col gap-2">
            <h2 className="text-sm font-semibold text-gray-100 uppercase tracking-wider">
              Attributes
            </h2>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Gender</span>
              <span className="text-gray-100 capitalize">
                {attributes.gender || "unknown"}
                {attributes.gender_confidence > 0 && (
                  <span className="ml-1 text-xs font-mono text-gray-500">
                    {(attributes.gender_confidence * 100).toFixed(0)}%
                  </span>
                )}
              </span>
            </div>
          </section>

          <h2 className="text-sm font-semibold text-gray-100 uppercase tracking-wider flex items-center gap-1.5">
            <Users size={13} /> Visually similar
          </h2>
          {similar.length === 0 ? (
            <p className="text-xs text-gray-500">None above the score threshold.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {similar.map((s) => (
                <li
                  key={s.person_id}
                  className="bg-panel border border-border rounded-lg p-2 flex gap-2 hover:border-accent"
                >
                  <Link
                    href={`/persons/${s.person_id}`}
                    className="flex gap-2 flex-1 min-w-0"
                  >
                    <div className="w-10 h-12 bg-black/40 rounded shrink-0 overflow-hidden flex items-center justify-center">
                      {s.person?.snapshot_url ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={s.person.snapshot_url}
                          alt={`Person ${s.person_id}`}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-gray-600 text-[10px]">—</span>
                      )}
                    </div>
                    <div className="flex flex-col justify-center min-w-0">
                      <span className="text-sm text-gray-100 font-semibold">
                        #{s.person_id}
                      </span>
                      <span className="text-xs text-gray-500 font-mono">
                        score {s.score.toFixed(3)}
                      </span>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </div>
  );
}
