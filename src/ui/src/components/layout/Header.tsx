"use client";

import { useEffect, useState } from "react";
import { LogOut, User } from "lucide-react";
import { getUser, logout } from "@/lib/auth";

export default function Header({ title }: { title?: string }) {
  const [username, setUsername] = useState<string>("");

  useEffect(() => {
    const u = getUser();
    if (u) setUsername(u.sub);
  }, []);

  return (
    <header className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0 bg-surface">
      <h1 className="text-sm font-semibold text-gray-100 tracking-tight">
        {title ?? "ReID Dashboard"}
      </h1>

      <div className="flex items-center gap-3">
        {username && (
          <span className="flex items-center gap-1.5 text-xs text-gray-400">
            <User size={13} />
            {username}
          </span>
        )}
        <button
          type="button"
          onClick={logout}
          className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-100 transition-colors"
          title="Logout"
        >
          <LogOut size={13} />
          Logout
        </button>
      </div>
    </header>
  );
}
