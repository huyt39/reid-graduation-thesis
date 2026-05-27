import { DashboardLayout } from "@/components/dashboard-layout";
import { StatsCards } from "./_components/stats-cards";
import { RecentPersons } from "./_components/recent-persons";
import { DeviceStatus } from "./_components/device-status";
import { ActivityByDevice } from "./_components/activity-by-device";

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <div className="space-y-8 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Overview</h2>
          <p className="text-sm text-muted-foreground">
            Live counts from the query service.
          </p>
        </div>

        <StatsCards />

        <div>
          <h3 className="text-base font-semibold tracking-tight mb-4">Recently Active Persons</h3>
          <RecentPersons />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <h3 className="text-base font-semibold tracking-tight mb-4">Device Status</h3>
            <DeviceStatus />
          </div>
          <div>
            <h3 className="text-base font-semibold tracking-tight mb-4">Activity</h3>
            <ActivityByDevice />
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
