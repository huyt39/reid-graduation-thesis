"use client";

import { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { PersonSnapshot } from "@/components/person-snapshot";
import { usePersons } from "@/hooks/use-persons";
import { formatRelative } from "@/lib/date-format";

export function PersonsList() {
  const [page, setPage] = useState(1);
  const pageSize = 20;
  const { data, isLoading, error } = usePersons({ page, page_size: pageSize });

  if (isLoading && !data) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-40" />
        ))}
      </div>
    );
  }

  if (error) {
    return <div className="text-sm text-destructive">Failed to load persons: {error.message}</div>;
  }

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  if (items.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-12 text-center text-sm text-muted-foreground">
        No persons found.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {items.map((person) => (
          <Link key={person.person_id} href={`/persons/${person.person_id}`}>
            <Card className="hover:border-primary/40 transition-colors h-full">
              <CardHeader className="pb-3">
                <PersonSnapshot
                  src={person.snapshot_url}
                  alt={`Person ${person.person_id} snapshot`}
                  label={`#${person.person_id}`}
                  className="mb-4 aspect-[4/5]"
                />
                <CardTitle className="flex items-center justify-between text-base">
                  <span>#{person.person_id}</span>
                  <Badge variant={person.is_active ? "default" : "secondary"}>
                    {person.is_active ? "Active" : "Inactive"}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Gender</span>
                  <span className="capitalize">{person.attributes.gender || "—"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Sightings</span>
                  <span>{person.stats.sighting_count.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Last seen</span>
                  <span>{formatRelative(person.stats.last_seen_at)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Last device</span>
                  <span className="font-mono text-xs">{person.stats.last_seen_device || "—"}</span>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <div className="flex items-center justify-between pt-2">
        <span className="text-sm text-muted-foreground">
          Page {page} of {totalPages} · {total.toLocaleString()} persons
        </span>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
