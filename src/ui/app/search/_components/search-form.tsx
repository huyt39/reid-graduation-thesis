"use client";

import { FormEvent, useMemo, useState } from "react";
import { Search, SendHorizontal } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PersonSnapshot } from "@/components/person-snapshot";
import { Skeleton } from "@/components/ui/skeleton";
import { useSearch } from "@/hooks/use-search";
import type { NLQueryResult, Person } from "@/types";

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

// Renders the natural-language answer as Markdown. No typography plugin is
// installed, so element spacing/lists are styled via arbitrary variants.
function AnswerBlock({ markdown }: { markdown: string }) {
  return (
    <div className="rounded-md border bg-muted/30 p-4 text-sm leading-relaxed [&_a]:underline [&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_li]:my-0.5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-2 [&_strong]:font-semibold [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
    </div>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPerson(value: unknown): value is Person {
  return isRecord(value) && typeof value.person_id === "number";
}

function extractPeople(result: NLQueryResult): Person[] {
  const payload = result.result;
  if (!isRecord(payload)) return [];

  if (isPerson(payload.person)) {
    return [payload.person];
  }

  if (Array.isArray(payload.items)) {
    return payload.items.filter(isPerson);
  }

  if (Array.isArray(payload.similar_persons)) {
    return payload.similar_persons
      .map((item) => (isRecord(item) ? item.person : null))
      .filter(isPerson);
  }

  return [];
}

function personAttributeSummary(person: Person): string {
  const labels = [
    person.attributes.gender,
    person.attributes.age_child,
    person.attributes.backpack,
    person.attributes.sidebag,
    person.attributes.hat,
    person.attributes.glasses,
    person.attributes.sleeve,
    person.attributes.lower,
  ].filter((value) => value && value !== "unknown");

  return labels.length > 0 ? labels.join(", ") : "No stable attributes";
}

function PeopleResults({ result }: { result: NLQueryResult }) {
  const people = extractPeople(result);
  if (people.length === 0) return null;

  return (
    <div className="rounded-md border p-3">
      <div className="text-muted-foreground text-xs font-medium uppercase tracking-wide">
        People
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {people.map((person) => (
          <div
            key={person.person_id}
            className="grid grid-cols-[88px_1fr] gap-3 rounded-md border bg-background p-2"
          >
            <PersonSnapshot
              src={person.snapshot_url}
              alt={`Person ${person.person_id} snapshot`}
              label={`#${person.person_id}`}
              className="aspect-[4/5] w-[88px] rounded-md"
              previewTitle={`Person #${person.person_id} snapshot`}
            />
            <div className="min-w-0 py-1">
              <div className="truncate text-sm font-medium">Person #{person.person_id}</div>
              <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                {personAttributeSummary(person)}
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                {person.stats?.sighting_count ?? 0} sightings
              </div>
              {person.stats?.last_seen_device ? (
                <div className="truncate text-xs text-muted-foreground">
                  Last seen at {person.stats.last_seen_device}
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </div>
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
                <AnswerBlock markdown={result.summary} />
              ) : (
                <p className="text-sm text-muted-foreground">
                  No answer was generated for this query. Open the details below to inspect the raw
                  result.
                </p>
              )}

              <PeopleResults result={result} />

              <details className="rounded-md border p-3">
                <summary className="text-muted-foreground cursor-pointer text-xs font-medium uppercase tracking-wide">
                  Details (parsed query &amp; raw result)
                </summary>
                <div className="mt-3 space-y-4">
                  <div>
                    <div className="mb-2 text-sm font-medium">Parsed query</div>
                    <JsonBlock value={result.parsed_query ?? null} />
                  </div>
                  <div>
                    <div className="mb-2 text-sm font-medium">Database result</div>
                    <JsonBlock value={result.result ?? result} />
                  </div>
                </div>
              </details>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
