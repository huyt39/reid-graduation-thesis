import { DashboardLayout } from "@/components/dashboard-layout";
import { PersonsList } from "./_components/persons-list";

export default function PersonsPage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Persons</h2>
          <p className="text-sm text-muted-foreground">
            Re-identified individuals tracked across the camera network.
          </p>
        </div>
        <PersonsList />
      </div>
    </DashboardLayout>
  );
}
