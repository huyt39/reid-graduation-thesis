import AppShell from "@/components/layout/AppShell";
import PersonDetail from "@/components/persons/PersonDetail";

export default function PersonDetailPage({ params }: { params: { id: string } }) {
  const id = Number.parseInt(params.id, 10);
  if (Number.isNaN(id)) {
    return (
      <AppShell title="Person">
        <p className="p-5 text-bad text-sm">Invalid person id.</p>
      </AppShell>
    );
  }
  return (
    <AppShell title={`Person #${id}`}>
      <PersonDetail personId={id} />
    </AppShell>
  );
}
