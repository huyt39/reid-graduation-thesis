import { DashboardLayout } from "@/components/dashboard-layout";
import { OcclusionDemo } from "./_components/occlusion-demo";

export default function OcclusionPage() {
  return (
    <DashboardLayout>
      <div className="space-y-6 p-4 md:p-6">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">Occlusion Evidence</h2>
          <p className="text-sm text-muted-foreground">
            Side-by-side ReID evidence for selected and rejected tracklet frames.
          </p>
        </div>
        <OcclusionDemo />
      </div>
    </DashboardLayout>
  );
}
