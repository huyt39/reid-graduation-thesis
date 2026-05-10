"use client";

import { useState } from "react";
import { Wifi, WifiOff, Loader2 } from "lucide-react";
import { useWebSocket } from "@/hooks/useWebSocket";
import DeviceSelector from "@/components/DeviceSelector";
import LiveFeed from "@/components/LiveFeed";
import PersonPanel from "@/components/PersonPanel";
import QueryPanel from "@/components/QueryPanel";

const WS_URL =
  (typeof window !== "undefined" && process.env.NEXT_PUBLIC_STREAMING_WS) ||
  "ws://localhost:8765";

function ConnectionBadge({ state }: { state: "connecting" | "connected" | "disconnected" }) {
  if (state === "connected") {
    return (
      <span className="flex items-center gap-1 text-xs text-good">
        <Wifi size={13} /> Live
      </span>
    );
  }
  if (state === "connecting") {
    return (
      <span className="flex items-center gap-1 text-xs text-mid">
        <Loader2 size={13} className="animate-spin" /> Connecting
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 text-xs text-bad">
      <WifiOff size={13} /> Disconnected
    </span>
  );
}

export default function DashboardPage() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);

  const { connectionState, deviceIds, framesByDevice } = useWebSocket(
    `${WS_URL}/ws`,
    selectedDevice,
  );

  const activeDevice = selectedDevice ?? deviceIds[0] ?? null;
  const currentFrame = activeDevice ? (framesByDevice[activeDevice] ?? null) : null;
  const persons = currentFrame?.tracked_persons ?? [];

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Sub-toolbar inside the AppShell main area */}
      <div className="flex items-center justify-between px-5 py-2 border-b border-border shrink-0 bg-panel/40">
        <DeviceSelector
          deviceIds={deviceIds}
          selected={activeDevice}
          onChange={(id) => setSelectedDevice(id)}
        />
        <ConnectionBadge state={connectionState} />
      </div>

      {/* Main body */}
      <div className="flex flex-1 gap-3 p-3 overflow-hidden">
        <PersonPanel persons={persons} />
        <div className="flex flex-1 flex-col gap-3 overflow-hidden">
          <LiveFeed frame={currentFrame} />
          <QueryPanel />
        </div>
      </div>
    </div>
  );
}
