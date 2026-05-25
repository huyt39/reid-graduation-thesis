"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { PersonSnapshot } from "@/components/person-snapshot";
import { usePersons } from "@/hooks/use-persons";
import { formatRelative } from "@/lib/date-format";

export function RecentPersons() {
  const { data, isLoading, error } = usePersons({ is_active: true, page: 1, page_size: 6 });

  if (isLoading && !data) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-52" />
        ))}
      </div>
    );
  }

  if (error) {
    return <div className="text-sm text-destructive">Failed to load active persons: {error.message}</div>;
  }

  const items = data?.items ?? [];

  if (items.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-12 text-center text-sm text-muted-foreground">
        No active persons right now.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
      {items.map((person) => (
        <Link key={person.person_id} href={`/persons/${person.person_id}`}>
          <Card className="hover:border-primary/40 transition-colors h-full">
            <CardHeader className="pb-2 px-3 pt-3">
              <PersonSnapshot
                src={person.snapshot_url}
                alt={`Person ${person.person_id}`}
                label={`#${person.person_id}`}
                className="aspect-[3/4]"
                previewTitle={`Person #${person.person_id}`}
              />
            </CardHeader>
            <CardContent className="px-3 pb-3 space-y-1 text-xs">
              <CardTitle className="text-sm">#{person.person_id}</CardTitle>
              <div className="text-muted-foreground">
                {formatRelative(person.stats.last_seen_at)}
              </div>
              <div className="text-muted-foreground truncate">
                {person.stats.last_seen_device || "—"}
              </div>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  );
}
