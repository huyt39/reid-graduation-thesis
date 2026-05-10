"use client";

import { useState } from "react";
import AppShell from "@/components/layout/AppShell";
import SearchBar from "@/components/search/SearchBar";
import SearchResults from "@/components/search/SearchResults";
import { naturalQuery } from "@/lib/api";
import type { NLQueryResult } from "@/types";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState<NLQueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function runQuery() {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await naturalQuery(query.trim());
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell title="Search">
      <div className="flex flex-col gap-4 p-5 max-w-5xl">
        <SearchBar
          query={query}
          onQueryChange={setQuery}
          onSubmit={runQuery}
          loading={loading}
        />
        <SearchResults data={data} rawError={error} />
      </div>
    </AppShell>
  );
}
