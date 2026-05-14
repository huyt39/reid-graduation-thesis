"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface TrackedPerson {
  person_id: number;
  bbox: [number, number, number, number];
  confidence: number;
  gender: string;
  gender_confidence: number;
  age_child?: string;
  age_child_confidence?: number;
  backpack?: string;
  backpack_confidence?: number;
  sidebag?: string;
  sidebag_confidence?: number;
  hat?: string;
  hat_confidence?: number;
  glasses?: string;
  glasses_confidence?: number;
  sleeve?: string;
  sleeve_confidence?: number;
  lower?: string;
  lower_confidence?: number;
  tracklet_id: string | null;
  tracklet_state: string | null;
  snapshot_url?: string | null;
  visibility_score: number;
  live_visibility_score: number;
  overlap_ratio: number;
  quality: {
    v_avg: number;
    embedding_consistency: number;
    overall_consistency: number;
    good_frame_ratio: number;
  } | null;
  matching: {
    method: string;
    source: string;
    similarity_score: number | null;
    runner_up_score: number | null;
    margin_to_runner_up: number | null;
    reuse_person_id: number | null;
    tentative_attempts: number | null;
    canonical_update_applied: boolean | null;
  } | null;
  attributes: Record<string, string> | null;
  status?: string | null;
}

export interface FrameUpdate {
  schema_version: number;
  source?: string;
  device_id: string;
  frame_number: number;
  tracked_persons: TrackedPerson[];
  created_at: number;
  image_base64: string;
}

export type ConnectionState = "connecting" | "connected" | "disconnected";

interface UseWebSocketResult {
  connectionState: ConnectionState;
  deviceIds: string[];
  currentFrame: FrameUpdate | null;
}

interface FrameUpdateMessage extends FrameUpdate {
  type: "frame_update";
}

const BACKOFF_CAP_MS = 8000;

function subscribeToDevice(ws: WebSocket | null, deviceId: string | null): void {
  if (!ws || ws.readyState !== WebSocket.OPEN || !deviceId) return;
  ws.send(JSON.stringify({ type: "subscribe_device", device_id: deviceId }));
}

export function useWebSocket(
  url: string | null,
  selectedDevice: string | null = null
): UseWebSocketResult {
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [deviceIds, setDeviceIds] = useState<string[]>([]);
  const [currentFrame, setCurrentFrame] = useState<FrameUpdate | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1000);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rafRef = useRef<number | null>(null);
  const unmountedRef = useRef(false);
  const framesByDeviceRef = useRef<Record<string, FrameUpdate>>({});
  const pendingFrameRef = useRef<FrameUpdate | null>(null);
  const activeDeviceRef = useRef<string | null>(selectedDevice);

  const flushPendingFrame = useCallback(() => {
    rafRef.current = null;
    if (pendingFrameRef.current) {
      setCurrentFrame(pendingFrameRef.current);
      pendingFrameRef.current = null;
    }
  }, []);

  const scheduleFrameUpdate = useCallback(
    (frame: FrameUpdate) => {
      pendingFrameRef.current = frame;
      if (rafRef.current !== null) return;
      rafRef.current = window.requestAnimationFrame(flushPendingFrame);
    },
    [flushPendingFrame]
  );

  const connect = useCallback(() => {
    if (!url || unmountedRef.current) return;

    setConnectionState("connecting");
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (unmountedRef.current) {
        ws.close();
        return;
      }
      setConnectionState("connected");
      backoffRef.current = 1000;
    };

    ws.onmessage = (ev) => {
      let msg: unknown;
      try {
        msg = JSON.parse(ev.data as string);
      } catch {
        return;
      }

      const m = msg as Record<string, unknown>;

      if (m.type === "device_list") {
        const maybeDevices = (m as { devices?: unknown }).devices;
        const devices = Array.isArray(maybeDevices)
          ? maybeDevices.filter((v): v is string => typeof v === "string")
          : [];
        setDeviceIds(devices);
        const nextActiveDevice = selectedDevice ?? devices[0] ?? null;
        activeDeviceRef.current = nextActiveDevice;
        if (nextActiveDevice) {
          const cachedFrame = framesByDeviceRef.current[nextActiveDevice] ?? null;
          pendingFrameRef.current = null;
          setCurrentFrame(cachedFrame);
        }
        subscribeToDevice(ws, nextActiveDevice);
        return;
      }

      if (m.type === "frame_update") {
        const frame = m as unknown as FrameUpdateMessage;
        framesByDeviceRef.current[frame.device_id] = frame;
        setDeviceIds((prev) =>
          prev.includes(frame.device_id) ? prev : [...prev, frame.device_id]
        );
        if (frame.device_id === activeDeviceRef.current) {
          scheduleFrameUpdate(frame);
        }
      }
    };

    ws.onclose = () => {
      if (unmountedRef.current) return;
      setConnectionState("disconnected");
      reconnectTimerRef.current = setTimeout(() => {
        backoffRef.current = Math.min(backoffRef.current * 2, BACKOFF_CAP_MS);
        connect();
      }, backoffRef.current);
    };

    ws.onerror = () => ws.close();
  }, [scheduleFrameUpdate, selectedDevice, url]);

  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (rafRef.current !== null) window.cancelAnimationFrame(rafRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  useEffect(() => {
    const nextActiveDevice = selectedDevice ?? deviceIds[0] ?? null;
    activeDeviceRef.current = nextActiveDevice;
    pendingFrameRef.current = null;
    setCurrentFrame(
      nextActiveDevice ? (framesByDeviceRef.current[nextActiveDevice] ?? null) : null
    );
    subscribeToDevice(wsRef.current, nextActiveDevice);
  }, [selectedDevice, deviceIds]);

  return { connectionState, deviceIds, currentFrame };
}
