"use client";

import { useEffect, useMemo, useState } from "react";
import { Pause, Play } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useWebSocket } from "@/hooks/use-websocket";
import { getStreamingWebSocketUrl } from "@/lib/streaming-websocket";
import { DeviceSelector, ALL_CAMERAS } from "./device-selector";
import { ConnectionBadge } from "./connection-badge";
import { LiveFeed } from "./live-feed";

const WS_URL = getStreamingWebSocketUrl();
// Raw video is served as MJPEG by the standalone raw_stream service (decoupled
// from the ReID path) and rendered natively by the browser for smoothness.
const RAW_STREAM_URL = process.env.NEXT_PUBLIC_RAW_STREAM_URL || "http://localhost:8770";

// Cap simultaneously-rendered feeds so a misconfigured fleet can't melt the
// browser. Multi-camera demo runs 2; raise if more cameras are wired up.
const MAX_FEEDS = 4;

export function LiveView() {
  // null = show every camera side-by-side; a value = focus a single camera.
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [isLiveActive, setIsLiveActive] = useState(true);
  const subscribedDeviceIds = useMemo(
    () => (selectedDevice ? [selectedDevice] : []),
    [selectedDevice]
  );

  // Processed stream is kept ONLY for the cross-camera badge (it is low-rate and
  // not the bottleneck). The raw video no longer flows through the WebSocket —
  // it comes from the MJPEG raw_stream service.
  const processed = useWebSocket(WS_URL, subscribedDeviceIds, {
    enabled: isLiveActive,
    maxFps: 10,
  });

  // The live feed device list comes from raw_stream's own /devices endpoint, NOT
  // the processed WebSocket stream. raw_stream opens its own VideoCapture per
  // device and is fully decoupled from the ReID pipeline, so live video shows
  // even when the reid_worker is down. Poll periodically to pick up new cameras.
  const [rawDeviceIds, setRawDeviceIds] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    const fetchDevices = async () => {
      try {
        const res = await fetch(`${RAW_STREAM_URL}/devices`);
        if (!res.ok) return;
        const data = (await res.json()) as { devices?: unknown };
        if (cancelled || !Array.isArray(data.devices)) return;
        const ids = data.devices.filter((d): d is string => typeof d === "string");
        setRawDeviceIds((prev) =>
          prev.length === ids.length && prev.every((id, i) => id === ids[i]) ? prev : ids
        );
      } catch {
        // raw_stream unreachable — keep the last known list (empty → "Waiting…").
      }
    };
    void fetchDevices();
    const interval = setInterval(() => void fetchDevices(), 8000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // Prefer raw_stream's device list; fall back to the processed stream only if
  // raw_stream is unreachable (preserves the old behavior in that case).
  const deviceIds = useMemo(() => {
    const source = rawDeviceIds.length > 0 ? rawDeviceIds : processed.deviceIds;
    return Array.from(new Set(source)).sort();
  }, [rawDeviceIds, processed.deviceIds]);

  // Cameras to render: the focused one, or all of them (capped).
  const visibleDeviceIds = useMemo(() => {
    if (selectedDevice && deviceIds.includes(selectedDevice)) return [selectedDevice];
    return deviceIds.slice(0, MAX_FEEDS);
  }, [deviceIds, selectedDevice]);

  // Cross-camera identities: a person_id seen on >= 2 devices (from the
  // low-rate processed stream). Pure badge signal; no per-frame overlay.
  const crossCameraIds = useMemo(() => {
    const devicesByPersonId = new Map<number, Set<string>>();
    for (const deviceId of visibleDeviceIds) {
      const frame = processed.framesByDevice[deviceId];
      if (!frame) continue;
      for (const person of frame.tracked_persons) {
        if (person.person_id != null) {
          const set = devicesByPersonId.get(person.person_id) ?? new Set<string>();
          set.add(deviceId);
          devicesByPersonId.set(person.person_id, set);
        }
      }
    }
    const xCam = new Set<number>();
    for (const [personId, devices] of devicesByPersonId) {
      if (devices.size >= 2) xCam.add(personId);
    }
    return xCam;
  }, [visibleDeviceIds, processed.framesByDevice]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <DeviceSelector
            deviceIds={deviceIds}
            selected={selectedDevice}
            onChange={(id) => setSelectedDevice(id === ALL_CAMERAS ? null : id)}
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
        <ConnectionBadge state={isLiveActive ? processed.connectionState : "disconnected"} />
      </div>

      <div
        className={
          visibleDeviceIds.length > 1
            ? "grid w-full gap-3 grid-cols-1 xl:grid-cols-2"
            : "flex w-full"
        }
      >
        {visibleDeviceIds.length === 0 ? (
          <LiveFeed deviceId={null} mjpegUrl={null} rawStreamUrl={null} isLiveActive={isLiveActive} />
        ) : (
          visibleDeviceIds.map((deviceId) => (
            <LiveFeed
              key={deviceId}
              deviceId={deviceId}
              mjpegUrl={`${RAW_STREAM_URL}/mjpeg?device_id=${encodeURIComponent(deviceId)}`}
              rawStreamUrl={RAW_STREAM_URL}
              isLiveActive={isLiveActive}
            />
          ))
        )}
      </div>
    </div>
  );
}
