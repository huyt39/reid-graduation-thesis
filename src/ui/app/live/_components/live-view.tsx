"use client";

import { useState } from "react";
import { useWebSocket } from "@/hooks/use-websocket";
import { DeviceSelector } from "./device-selector";
import { ConnectionBadge } from "./connection-badge";
import { LiveFeed } from "./live-feed";
import { PersonsPanel } from "./persons-panel";

const WS_URL = process.env.NEXT_PUBLIC_STREAMING_WS || "ws://localhost:8765";

export function LiveView() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);

  const { connectionState, deviceIds, framesByDevice } = useWebSocket(
    `${WS_URL}/ws`,
    selectedDevice
  );

  const activeDevice = selectedDevice ?? deviceIds[0] ?? null;
  const currentFrame = activeDevice ? (framesByDevice[activeDevice] ?? null) : null;
  const persons = currentFrame?.tracked_persons ?? [];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <DeviceSelector
          deviceIds={deviceIds}
          selected={activeDevice}
          onChange={setSelectedDevice}
        />
        <ConnectionBadge state={connectionState} />
      </div>

      <div className="flex flex-col lg:flex-row gap-4">
        <PersonsPanel persons={persons} />
        <LiveFeed frame={currentFrame} />
      </div>
    </div>
  );
}
