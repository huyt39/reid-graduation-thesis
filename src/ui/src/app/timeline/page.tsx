"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import AppShell from "@/components/layout/AppShell";
import TimelineView from "@/components/timeline/TimelineView";
import TimelinePersonPicker from "@/components/timeline/TimelinePersonPicker";

function TimelineInner() {
  const router = useRouter();
  const sp = useSearchParams();
  const raw = sp.get("person_id");
  const parsed = raw === null ? null : Number.parseInt(raw, 10);
  const initialId = parsed !== null && !Number.isNaN(parsed) ? parsed : null;

  const [personId, setPersonId] = useState<number | null>(initialId);

  useEffect(() => {
    setPersonId(initialId);
  }, [initialId]);

  const handleSelect = useCallback(
    (id: number | null) => {
      setPersonId(id);
      const next = id === null ? "/timeline" : `/timeline?person_id=${id}`;
      router.replace(next);
    },
    [router],
  );

  return (
    <div className="p-5 max-w-4xl flex flex-col gap-5">
      <TimelinePersonPicker selectedId={personId} onSelect={handleSelect} />
      <TimelineView personId={personId} />
    </div>
  );
}

export default function TimelinePage() {
  return (
    <AppShell title="Timeline">
      <Suspense fallback={<div className="p-5 text-gray-500 text-sm">Loading…</div>}>
        <TimelineInner />
      </Suspense>
    </AppShell>
  );
}
