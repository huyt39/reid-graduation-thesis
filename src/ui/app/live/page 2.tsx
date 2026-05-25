import { DashboardLayout } from "@/components/dashboard-layout";
import { LiveView } from "./_components/live-view";

export default function LivePage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Live</h2>
          <p className="text-sm text-muted-foreground">
            Live camera feed with tracked-person overlays from the streaming
            service.
          </p>
        </div>
        <LiveView />
      </div>
    </DashboardLayout>
  );
}
