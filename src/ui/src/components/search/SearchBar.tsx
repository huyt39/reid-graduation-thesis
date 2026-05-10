"use client";

import { Loader2, Send } from "lucide-react";

interface Props {
  query: string;
  onQueryChange: (s: string) => void;
  onSubmit: () => void;
  loading: boolean;
  examples?: string[];
}

export default function SearchBar({
  query,
  onQueryChange,
  onSubmit,
  loading,
  examples = [
    "show me person 42",
    "find all women",
    "where was person 100 today?",
    "list all cameras",
    "people similar to person 7",
    "how many times did person 5 appear by hour",
  ],
}: Props) {
  return (
    <div className="flex flex-col gap-2">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!loading && query.trim()) onSubmit();
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="Ask in plain English…"
          className="flex-1 bg-panel border border-border rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-accent"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="bg-accent hover:bg-blue-500 disabled:opacity-40 text-white rounded-lg px-4 py-2 flex items-center gap-1.5 text-sm transition-colors"
        >
          {loading ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <Send size={15} />
          )}
          Send
        </button>
      </form>

      <div className="flex flex-wrap gap-1.5">
        {examples.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => onQueryChange(ex)}
            className="text-xs text-gray-400 hover:text-gray-100 bg-panel/50 border border-border hover:border-accent rounded-full px-2.5 py-1 transition-colors"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
