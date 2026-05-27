"use client";

import { useMemo, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useWebSocket } from "@/hooks/use-websocket";
import {
  getCachedLiveIdentities,
  mergeRawWithProcessedIds,
  updateLiveIdentityCache,
  type LiveIdentityCacheEntry,
} from "@/lib/match-bboxes";
import { DeviceSelector } from "./device-selector";
import { ConnectionBadge } from "./connection-badge";
import { LiveFeed } from "./live-feed";
import { PersonsPanel } from "./persons-panel";

const WS_URL = process.env.NEXT_PUBLIC_STREAMING_WS || "ws://localhost:8765";

export function LiveView() {
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [isLiveActive, setIsLiveActive] = useState(true);
  const identityCacheRef = useRef(new Map<string, LiveIdentityCacheEntry>());

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
  const hybridTrackedPersons = useMemo(() => {
    const now = Date.now();
    if (processedFrame) {
      updateLiveIdentityCache(
        identityCacheRef.current,
        processedFrame.tracked_persons,
        processedFrame.frame_number,
        now
      );
    }
    if (!rawFrame || !processedFrame) return baseFrame?.tracked_persons ?? [];

    const currentKeys = new Set(
      processedFrame.tracked_persons
        .map((person) => person.live_track_key ?? person.tracklet_id)
        .filter((key): key is string => typeof key === "string")
    );
    const cachedPersons = getCachedLiveIdentities(
      identityCacheRef.current,
      rawFrame.frame_number,
      now
    ).filter((person) => {
      const key = person.live_track_key ?? person.tracklet_id;
      return !key || !currentKeys.has(key);
    });

    return mergeRawWithProcessedIds(rawFrame.tracked_persons, processedFrame.tracked_persons, {
      cachedPersons,
      minIou: 0.35,
      sourceSize: {
        width: processedFrame.image_width,
        height: processedFrame.image_height,
      },
      targetSize: {
        width: rawFrame.image_width,
        height: rawFrame.image_height,
      },
    });
  }, [baseFrame?.tracked_persons, processedFrame, rawFrame]);
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

      <div className="flex flex-col lg:flex-row gap-4 lg:items-start">
        <PersonsPanel persons={isLiveActive ? persons : []} />
        <LiveFeed frame={currentFrame} isLiveActive={isLiveActive} />
      </div>
    </div>
  );
}
