"use client";

import { useMemo, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useWebSocket, type FrameUpdate, type TrackedPerson } from "@/hooks/use-websocket";
import {
  getCachedLiveIdentities,
  mergeRawWithProcessedIds,
  updateLiveIdentityCache,
  type LiveIdentityCacheEntry,
} from "@/lib/match-bboxes";
import { DeviceSelector } from "./device-selector";
import { ConnectionBadge } from "./connection-badge";
import { LiveFeed } from "./live-feed";
import { PersonsPanel, type PanelPerson } from "./persons-panel";

const WS_URL = process.env.NEXT_PUBLIC_STREAMING_WS || "ws://localhost:8765";

// Cap simultaneously-rendered feeds so a misconfigured fleet can't melt the
// browser. Multi-camera demo runs 2; raise if more cameras are wired up.
const MAX_FEEDS = 4;

interface FeedView {
  deviceId: string;
  frame: FrameUpdate | null;
  usingRawFallback: boolean;
  idLagFrames: number;
}

export function LiveView() {
  // null = show every camera side-by-side; a value = focus a single camera.
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

  const deviceIds = useMemo(
    () => Array.from(new Set([...processed.deviceIds, ...raw.deviceIds])).sort(),
    [processed.deviceIds, raw.deviceIds]
  );

  // Cameras to render: the focused one, or all of them (capped).
  const visibleDeviceIds = useMemo(() => {
    if (selectedDevice && deviceIds.includes(selectedDevice)) return [selectedDevice];
    return deviceIds.slice(0, MAX_FEEDS);
  }, [deviceIds, selectedDevice]);

  // Build one hybrid frame (raw image + processed person_ids/attributes) per
  // visible camera. Identity-cache keys are `device_id:track_id` (namespaced
  // by the worker), so a single shared cache map is collision-free across
  // cameras. See live-view single-stream notes — same logic, applied per device.
  const { feeds, panelPersons, crossCameraIds } = useMemo(() => {
    const now = Date.now();
    const builtFeeds: FeedView[] = [];
    const allPanelPersons: PanelPerson[] = [];
    const devicesByPersonId = new Map<number, Set<string>>();

    for (const deviceId of visibleDeviceIds) {
      const processedFrame = processed.framesByDevice[deviceId] ?? null;
      const rawFrame = raw.framesByDevice[deviceId] ?? null;

      if (processedFrame) {
        updateLiveIdentityCache(
          identityCacheRef.current,
          processedFrame.tracked_persons,
          processedFrame.frame_number,
          now
        );
      }

      // Prefer the raw edge stream for the displayed image (stays smooth even
      // when the worker is CPU-bound); overlay processed IDs via IoU projection.
      const baseFrame = rawFrame ?? processedFrame;
      let hybridPersons: TrackedPerson[] = baseFrame?.tracked_persons ?? [];

      if (rawFrame && processedFrame) {
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
        hybridPersons = mergeRawWithProcessedIds(
          rawFrame.tracked_persons,
          processedFrame.tracked_persons,
          {
            cachedPersons,
            minIou: 0.35,
            sourceSize: { width: processedFrame.image_width, height: processedFrame.image_height },
            targetSize: { width: rawFrame.image_width, height: rawFrame.image_height },
          }
        );
      }

      const frame = baseFrame ? { ...baseFrame, tracked_persons: hybridPersons } : null;
      builtFeeds.push({
        deviceId,
        frame,
        usingRawFallback: !!rawFrame && !!processedFrame,
        idLagFrames:
          rawFrame && processedFrame
            ? Math.max(0, rawFrame.frame_number - processedFrame.frame_number)
            : 0,
      });

      // Panel uses pure processed persons (real IDs + attribute hysteresis),
      // tagged with their camera so we can badge + spot cross-camera identities.
      const sourcePersons = processedFrame?.tracked_persons ?? hybridPersons;
      for (const person of sourcePersons) {
        allPanelPersons.push({ ...person, deviceId });
        if (person.person_id != null) {
          const set = devicesByPersonId.get(person.person_id) ?? new Set<string>();
          set.add(deviceId);
          devicesByPersonId.set(person.person_id, set);
        }
      }
    }

    const xCamIds = new Set<number>();
    for (const [personId, devices] of devicesByPersonId) {
      if (devices.size >= 2) xCamIds.add(personId);
    }

    return { feeds: builtFeeds, panelPersons: allPanelPersons, crossCameraIds: xCamIds };
  }, [visibleDeviceIds, processed.framesByDevice, raw.framesByDevice]);

  const anyRawFallback = feeds.some((f) => f.usingRawFallback);
  const maxIdLag = feeds.reduce((m, f) => Math.max(m, f.idLagFrames), 0);
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
            selected={selectedDevice}
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
          {deviceIds.length > 1 ? (
            <Badge variant="secondary">{deviceIds.length} cameras</Badge>
          ) : null}
          {crossCameraIds.size > 0 ? (
            <Badge className="bg-fuchsia-600 text-white hover:bg-fuchsia-600">
              {crossCameraIds.size} cross-camera{crossCameraIds.size > 1 ? " IDs" : " ID"}
            </Badge>
          ) : null}
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

      {anyRawFallback ? (
        <div className="flex items-center gap-2">
          <Badge variant="secondary">Live • smooth raw</Badge>
          <p className="text-sm text-muted-foreground">
            Live video uses the raw edge stream for smoothness. Person IDs + attributes from the
            worker resolve asynchronously
            {maxIdLag > 0 ? ` (up to ${maxIdLag} frames behind)` : null}.
          </p>
        </div>
      ) : null}

      <div className="flex flex-col lg:flex-row gap-4 lg:items-start">
        <PersonsPanel
          persons={isLiveActive ? panelPersons : []}
          crossCameraIds={crossCameraIds}
          showDeviceBadge={deviceIds.length > 1}
        />
        <div
          className={
            feeds.length > 1
              ? "grid flex-1 gap-3 grid-cols-1 xl:grid-cols-2"
              : "flex flex-1"
          }
        >
          {feeds.length === 0 ? (
            <LiveFeed frame={null} isLiveActive={isLiveActive} />
          ) : (
            feeds.map((f) => (
              <LiveFeed key={f.deviceId} frame={f.frame} isLiveActive={isLiveActive} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}
