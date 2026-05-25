import { DashboardLayout } from "@/components/dashboard-layout";
import { SearchForm } from "./_components/search-form";

export default function SearchPage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Search</h2>
          <p className="text-sm text-muted-foreground">
            Ask the ReID query service in natural language and inspect the parsed database query.
          </p>
        </div>
        <SearchForm />
      </div>
    </DashboardLayout>
  );
}
