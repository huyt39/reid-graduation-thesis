"use client";

import { useState } from "react";
import Link from "next/link";
import { Send, Loader2, ExternalLink } from "lucide-react";
import { naturalQuery } from "@/lib/api";

interface NLQueryWrapper {
  parsed_query?: { query_type: string; params?: Record<string, unknown> };
  result?: unknown;
  message?: string;
}

export default function QueryPanel() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<NLQueryWrapper | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await naturalQuery(q);
      setResult(data as NLQueryWrapper);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-2 border-t border-border pt-3">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about persons… e.g. Who wore a red shirt near entrance?"
          className="flex-1 bg-panel border border-border rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-accent"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="bg-accent hover:bg-blue-500 disabled:opacity-40 text-white rounded-lg px-3 py-2 flex items-center gap-1.5 text-sm transition-colors"
        >
          {loading ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <Send size={15} />
          )}
          Send
        </button>
      </form>

      {error && (
        <p className="text-bad text-xs px-1">{error}</p>
      )}

      {result && (
        <div className="bg-panel border border-border rounded-lg px-3 py-2 text-sm text-gray-300 max-h-40 overflow-y-auto flex flex-col gap-1.5">
          {result.parsed_query && (
            <div className="text-xs text-gray-500 font-mono">
              parsed:{" "}
              <span className="text-accent">{result.parsed_query.query_type}</span>{" "}
              {JSON.stringify(result.parsed_query.params ?? {})}
            </div>
          )}
          <pre className="text-xs text-gray-400 whitespace-pre-wrap break-words">
            {JSON.stringify(result.result ?? result, null, 2).slice(0, 1000)}
          </pre>
          <Link
            href={`/search?q=${encodeURIComponent(query)}`}
            className="text-xs text-accent hover:underline flex items-center gap-1"
          >
            <ExternalLink size={11} /> open in full search
          </Link>
        </div>
      )}
    </div>
  );
}
