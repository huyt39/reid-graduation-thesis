import { DashboardLayout } from "@/components/dashboard-layout";
import { SearchForm } from "./_components/search-form";

export default function SearchPage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Search</h2>
          <p className="text-sm text-muted-foreground">
            Run structured queries against the ReID query service.
          </p>
        </div>
        <SearchForm />
      </div>
    </DashboardLayout>
  );
}
