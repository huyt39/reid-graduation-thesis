import { DashboardLayout } from "@/components/dashboard-layout";
import { DevicesList } from "./_components/devices-list";

export default function DevicesPage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Devices</h2>
          <p className="text-sm text-muted-foreground">
            Cameras and sensors registered with the ReID pipeline.
          </p>
        </div>
        <DevicesList />
      </div>
    </DashboardLayout>
  );
}
