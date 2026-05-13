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
  quality: {
    v_avg: number;
    embedding_consistency: number;
    overall_consistency: number;
    good_frame_ratio: number;
  } | null;
  attributes: Record<string, string> | null;
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
  framesByDevice: Record<string, FrameUpdate>;
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
  const [framesByDevice, setFramesByDevice] = useState<Record<string, FrameUpdate>>({});

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1000);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);

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
        subscribeToDevice(ws, selectedDevice ?? devices[0] ?? null);
        return;
      }

      if (m.type === "frame_update") {
        const frame = m as unknown as FrameUpdateMessage;
        setFramesByDevice((prev) => ({ ...prev, [frame.device_id]: frame }));
        setDeviceIds((prev) =>
          prev.includes(frame.device_id) ? prev : [...prev, frame.device_id]
        );
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
  }, [url, selectedDevice]);

  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  useEffect(() => {
    subscribeToDevice(wsRef.current, selectedDevice ?? deviceIds[0] ?? null);
  }, [selectedDevice, deviceIds]);

  return { connectionState, deviceIds, framesByDevice };
}
