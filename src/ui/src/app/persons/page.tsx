import AppShell from "@/components/layout/AppShell";
import PersonList from "@/components/persons/PersonList";

export default function PersonsPage() {
  return (
    <AppShell title="Persons">
      <PersonList />
    </AppShell>
  );
}
