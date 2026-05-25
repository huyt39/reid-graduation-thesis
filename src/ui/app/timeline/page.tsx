import { DashboardLayout } from "@/components/dashboard-layout";
import { TimelineView } from "./_components/timeline-view";

export default function TimelinePage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Timeline</h2>
          <p className="text-sm text-muted-foreground">
            Audit a person&apos;s movement events across devices.
          </p>
        </div>
        <TimelineView />
      </div>
    </DashboardLayout>
  );
}
