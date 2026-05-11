import { DashboardLayout } from "@/components/dashboard-layout";
import { StatsCards } from "./_components/stats-cards";

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Overview</h2>
          <p className="text-sm text-muted-foreground">
            Live counts from the query service.
          </p>
        </div>
        <StatsCards />
      </div>
    </DashboardLayout>
  );
}
