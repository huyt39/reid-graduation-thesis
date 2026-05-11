import Link from "next/link";
import { ChevronLeft } from "lucide-react";
import { DashboardLayout } from "@/components/dashboard-layout";
import { PersonDetail } from "./_components/person-detail";

interface PageProps {
  params: Promise<{ personId: string }>;
}

export default async function PersonDetailPage({ params }: PageProps) {
  const { personId } = await params;
  const id = Number(personId);

  return (
    <DashboardLayout>
      <div className="space-y-4 p-4 md:p-6">
        <Link
          href="/persons"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          Back to persons
        </Link>
        {Number.isFinite(id) ? (
          <PersonDetail personId={id} />
        ) : (
          <p className="text-destructive text-sm">Invalid person id.</p>
        )}
      </div>
    </DashboardLayout>
  );
}
