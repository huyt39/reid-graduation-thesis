"use client";

import Sidebar from "./Sidebar";
import Header from "./Header";

interface Props {
  title?: string;
  children: React.ReactNode;
}

export default function AppShell({ title, children }: Props) {
  return (
    <div className="h-screen flex bg-surface overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title={title} />
        <main className="flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  );
}
