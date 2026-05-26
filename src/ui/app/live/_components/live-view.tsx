"use client";

import { useState } from "react";
import { Pause, Play } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useWebSocket } from "@/hooks/use-websocket";
import { mergeRawWithProcessedIds } from "@/lib/match-bboxes";
import { DeviceSelector } from "./device-selector";
import { ConnectionBadge } from "./connection-badge";
import { LiveFeed } from "./live-feed";
import { PersonsPanel } from "./persons-panel";

const WS_URL = process.env.NEXT_PUBLIC_STREAMING_WS || "ws://localhost:8765";

export function LiveView() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [isLiveActive, setIsLiveActive] = useState(true);

  const processed = useWebSocket(`${WS_URL}/ws`, selectedDevice, {
    enabled: isLiveActive,
    maxFps: 30,
  });
  const raw = useWebSocket(`${WS_URL}/ws/raw`, selectedDevice, {
    enabled: isLiveActive,
    maxFps: 30,
  });

  const deviceIds = Array.from(new Set([...processed.deviceIds, ...raw.deviceIds]));
  const activeDevice = selectedDevice ?? processed.deviceIds[0] ?? raw.deviceIds[0] ?? null;
  const processedFrame = activeDevice ? (processed.framesByDevice[activeDevice] ?? null) : null;
  const rawFrame = activeDevice ? (raw.framesByDevice[activeDevice] ?? null) : null;
  // Prefer the raw edge stream for the live image so the video stays smooth even when
  // the worker is CPU-bound and lags. The persons panel still uses processed data so
  // we get person_ids + attributes asynchronously as they resolve.
  const baseFrame = rawFrame ?? processedFrame;
  const usingRawFallback = !!rawFrame && !!processedFrame;
  // Hybrid: raw bboxes (match the image) + processed person_ids/attributes
  // (real IDs) via IoU projection. PersonsPanel keeps using pure processed
  // data so attribute hysteresis isn't affected.
  const hybridTrackedPersons =
    rawFrame && processedFrame
      ? mergeRawWithProcessedIds(rawFrame.tracked_persons, processedFrame.tracked_persons)
      : (baseFrame?.tracked_persons ?? []);
  const currentFrame = baseFrame ? { ...baseFrame, tracked_persons: hybridTrackedPersons } : null;
  const persons = currentFrame?.tracked_persons ?? [];
  const idLagFrames =
    rawFrame && processedFrame
      ? Math.max(0, rawFrame.frame_number - processedFrame.frame_number)
      : 0;
  const connectionState =
    processed.connectionState === "connected" || raw.connectionState === "connected"
      ? "connected"
      : processed.connectionState === "connecting" || raw.connectionState === "connecting"
        ? "connecting"
        : "disconnected";

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <DeviceSelector
            deviceIds={deviceIds}
            selected={activeDevice}
            onChange={setSelectedDevice}
          />
          <Button
            type="button"
            variant={isLiveActive ? "secondary" : "default"}
            onClick={() => setIsLiveActive((prev) => !prev)}
          >
            {isLiveActive ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isLiveActive ? "Pause live" : "Start live"}
          </Button>
        </div>
        <ConnectionBadge state={isLiveActive ? connectionState : "disconnected"} />
      </div>

      {!isLiveActive ? (
        <div className="flex items-center gap-2">
          <Badge variant="outline">Paused</Badge>
          <p className="text-sm text-muted-foreground">
            Live stream is paused. Press Start live to resume.
          </p>
        </div>
      ) : null}

      {usingRawFallback ? (
        <div className="flex items-center gap-2">
          <Badge variant="secondary">Live • smooth raw</Badge>
          <p className="text-sm text-muted-foreground">
            Live video uses the raw edge stream for smoothness. Person IDs + attributes from the
            worker resolve asynchronously
            {idLagFrames > 0 ? ` (currently ${idLagFrames} frames behind)` : null}.
          </p>
        </div>
      ) : null}

      <div className="flex flex-col lg:flex-row gap-4">
        <PersonsPanel persons={isLiveActive ? persons : []} />
        <LiveFeed frame={currentFrame} isLiveActive={isLiveActive} />
      </div>
    </div>
  );
}
