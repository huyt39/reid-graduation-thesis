"use client";

import { FormEvent, useMemo, useState } from "react";
import { Search, SendHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useSearch } from "@/hooks/use-search";

const EXAMPLE_QUERIES = [
  "show me person 42",
  "find all women last seen at camera-1",
  "how many times did person 5 appear by hour",
  "list all cameras",
];

function JsonBlock({ value }: { value: unknown }) {
  const text = useMemo(() => JSON.stringify(value ?? null, null, 2), [value]);
  return (
    <pre className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/50 p-3 text-sm leading-relaxed font-sans">
      {text}
    </pre>
  );
}

export function SearchForm() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const { result, isLoading, error, runNatural } = useSearch();

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleaned = query.trim();
    if (!cleaned || isLoading) return;
    setSubmittedQuery(cleaned);
    await runNatural(cleaned);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(360px,480px)_1fr]">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Search className="h-4 w-4" />
            Natural query
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4">
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Ask about persons, cameras, timelines, or sightings..."
              rows={5}
              className="border-input bg-background ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring min-h-32 w-full resize-none rounded-md border px-3 py-2 text-sm outline-none transition-[color,box-shadow] focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
            <Button
              type="submit"
              disabled={isLoading || query.trim().length === 0}
              className="w-full"
            >
              <SendHorizontal className="h-4 w-4" />
              {isLoading ? "Running..." : "Run query"}
            </Button>
          </form>

          <div className="mt-5 space-y-2">
            {EXAMPLE_QUERIES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setQuery(example)}
                className="hover:bg-muted block w-full rounded-md border px-3 py-2 text-left text-sm transition-colors"
              >
                {example}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Result</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {isLoading ? (
            <Skeleton className="h-48 w-full" />
          ) : error ? (
            <p className="text-destructive text-sm">{error}</p>
          ) : !result ? (
            <p className="text-sm text-muted-foreground">
              Enter a natural-language query and run it to inspect the parsed query and database
              result.
            </p>
          ) : (
            <>
              <div className="rounded-md border p-3">
                <div className="text-muted-foreground text-xs font-medium uppercase tracking-wide">
                  Query
                </div>
                <div className="mt-1 text-sm">{submittedQuery}</div>
              </div>

              {result.summary ? (
                <div className="rounded-md border p-3">
                  <div className="text-muted-foreground text-xs font-medium uppercase tracking-wide">
                    Summary
                  </div>
                  <p className="mt-1 text-sm leading-relaxed">{result.summary}</p>
                </div>
              ) : null}

              <div>
                <div className="mb-2 text-sm font-medium">Parsed query</div>
                <JsonBlock value={result.parsed_query ?? null} />
              </div>

              <div>
                <div className="mb-2 text-sm font-medium">Database result</div>
                <JsonBlock value={result.result ?? result} />
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
