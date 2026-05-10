"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, Users, Search, Clock } from "lucide-react";

const NAV = [
  { href: "/dashboard", label: "Live View", icon: Activity },
  { href: "/persons", label: "Persons", icon: Users },
  { href: "/search", label: "Search", icon: Search },
  { href: "/timeline", label: "Timeline", icon: Clock },
] as const;

export default function Sidebar() {
  const pathname = usePathname() || "";

  return (
    <aside className="w-48 shrink-0 bg-panel border-r border-border flex flex-col py-3">
      <div className="px-4 pb-4 text-sm font-semibold text-gray-100 tracking-tight">
        ReID
      </div>

      <nav className="flex flex-col gap-0.5 px-2">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={[
                "flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors",
                active
                  ? "bg-accent/10 text-accent"
                  : "text-gray-400 hover:bg-surface hover:text-gray-100",
              ].join(" ")}
            >
              <Icon size={15} />
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
