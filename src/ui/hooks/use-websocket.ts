"use client";

import { startTransition, useCallback, useEffect, useRef, useState } from "react";

export interface TrackedPerson {
  person_id: number | null;
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
  track_id: number | null;
  live_track_key: string | null;
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
  image_width: number | null;
  image_height: number | null;
}

export type ConnectionState = "connecting" | "connected" | "disconnected";

interface UseWebSocketResult {
  connectionState: ConnectionState;
  deviceIds: string[];
  framesByDevice: Record<string, FrameUpdate>;
}

interface UseWebSocketOptions {
  enabled?: boolean;
  maxFps?: number;
}

const BACKOFF_CAP_MS = 8000;

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asOptionalString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function normalizeTrackedPerson(raw: unknown): TrackedPerson {
  const person = (raw ?? {}) as Record<string, unknown>;
  const rawBbox = Array.isArray(person.bbox) ? person.bbox : [];
  const bbox: [number, number, number, number] = [
    asNumber(rawBbox[0]),
    asNumber(rawBbox[1]),
    asNumber(rawBbox[2]),
    asNumber(rawBbox[3]),
  ];
  const qualityRaw =
    person.quality && typeof person.quality === "object"
      ? (person.quality as Record<string, unknown>)
      : null;
  const matchingRaw =
    person.matching && typeof person.matching === "object"
      ? (person.matching as Record<string, unknown>)
      : null;

  return {
    person_id: typeof person.person_id === "number" ? person.person_id : null,
    bbox,
    confidence: asNumber(person.confidence),
    gender: asString(person.gender, "unknown"),
    gender_confidence: asNumber(person.gender_confidence),
    age_child: asOptionalString(person.age_child) ?? undefined,
    age_child_confidence: asNumber(person.age_child_confidence),
    backpack: asOptionalString(person.backpack) ?? undefined,
    backpack_confidence: asNumber(person.backpack_confidence),
    sidebag: asOptionalString(person.sidebag) ?? undefined,
    sidebag_confidence: asNumber(person.sidebag_confidence),
    hat: asOptionalString(person.hat) ?? undefined,
    hat_confidence: asNumber(person.hat_confidence),
    glasses: asOptionalString(person.glasses) ?? undefined,
    glasses_confidence: asNumber(person.glasses_confidence),
    sleeve: asOptionalString(person.sleeve) ?? undefined,
    sleeve_confidence: asNumber(person.sleeve_confidence),
    lower: asOptionalString(person.lower) ?? undefined,
    lower_confidence: asNumber(person.lower_confidence),
    track_id: typeof person.track_id === "number" ? person.track_id : null,
    live_track_key: asOptionalString(person.live_track_key),
    tracklet_id: asOptionalString(person.tracklet_id),
    tracklet_state: asOptionalString(person.tracklet_state),
    snapshot_url: asOptionalString(person.snapshot_url),
    visibility_score: asNumber(person.visibility_score),
    live_visibility_score: asNumber(
      person.live_visibility_score,
      asNumber(person.visibility_score)
    ),
    overlap_ratio: asNumber(person.overlap_ratio),
    quality: qualityRaw
      ? {
          v_avg: asNumber(qualityRaw.v_avg),
          embedding_consistency: asNumber(qualityRaw.embedding_consistency),
          overall_consistency: asNumber(qualityRaw.overall_consistency),
          good_frame_ratio: asNumber(qualityRaw.good_frame_ratio),
        }
      : null,
    matching: matchingRaw
      ? {
          method: asString(matchingRaw.method),
          source: asString(matchingRaw.source),
          similarity_score:
            typeof matchingRaw.similarity_score === "number" ? matchingRaw.similarity_score : null,
          runner_up_score:
            typeof matchingRaw.runner_up_score === "number" ? matchingRaw.runner_up_score : null,
          margin_to_runner_up:
            typeof matchingRaw.margin_to_runner_up === "number"
              ? matchingRaw.margin_to_runner_up
              : null,
          reuse_person_id:
            typeof matchingRaw.reuse_person_id === "number" ? matchingRaw.reuse_person_id : null,
          tentative_attempts:
            typeof matchingRaw.tentative_attempts === "number"
              ? matchingRaw.tentative_attempts
              : null,
          canonical_update_applied:
            typeof matchingRaw.canonical_update_applied === "boolean"
              ? matchingRaw.canonical_update_applied
              : null,
        }
      : null,
    attributes:
      person.attributes && typeof person.attributes === "object"
        ? (person.attributes as Record<string, string>)
        : null,
    status: asOptionalString(person.status),
  };
}

function normalizeFrameUpdate(raw: Record<string, unknown>): FrameUpdate {
  const trackedPersonsRaw = Array.isArray(raw.tracked_persons) ? raw.tracked_persons : [];
  return {
    schema_version: asNumber(raw.schema_version, 2),
    source: asOptionalString(raw.source) ?? undefined,
    device_id: asString(raw.device_id),
    frame_number: asNumber(raw.frame_number),
    tracked_persons: trackedPersonsRaw.map(normalizeTrackedPerson),
    created_at: asNumber(raw.created_at),
    image_base64: asString(raw.image_base64),
    image_width: typeof raw.image_width === "number" ? raw.image_width : null,
    image_height: typeof raw.image_height === "number" ? raw.image_height : null,
  };
}

function subscribeToDevices(ws: WebSocket | null, deviceIds: string[]): void {
  if (!ws || ws.readyState !== WebSocket.OPEN || deviceIds.length === 0) return;
  if (deviceIds.length === 1) {
    ws.send(JSON.stringify({ type: "subscribe_device", device_id: deviceIds[0] }));
    return;
  }
  ws.send(JSON.stringify({ type: "subscribe_devices", device_ids: deviceIds }));
}

export function useWebSocket(
  url: string | null,
  subscribedDeviceIds: string[] = [],
  options: UseWebSocketOptions = {}
): UseWebSocketResult {
  const { enabled = true, maxFps = 30 } = options;
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [deviceIds, setDeviceIds] = useState<string[]>([]);
  const [framesByDevice, setFramesByDevice] = useState<Record<string, FrameUpdate>>({});

  const wsRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef(1000);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);
  const subscribedDeviceIdsRef = useRef<string[]>(subscribedDeviceIds);
  const pendingFramesRef = useRef<Record<string, FrameUpdate>>({});
  const pendingDeviceIdsRef = useRef<Set<string>>(new Set());
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastFlushAtRef = useRef(0);

  const flushPending = useCallback(() => {
    flushTimerRef.current = null;
    const nextFrames = pendingFramesRef.current;
    const nextDeviceIds = Array.from(pendingDeviceIdsRef.current);
    pendingFramesRef.current = {};

    lastFlushAtRef.current = Date.now();
    startTransition(() => {
      if (Object.keys(nextFrames).length > 0) {
        setFramesByDevice((prev) => ({ ...prev, ...nextFrames }));
      }
      if (nextDeviceIds.length > 0) {
        setDeviceIds(nextDeviceIds);
      }
    });
  }, []);

  const scheduleFlush = useCallback(() => {
    if (flushTimerRef.current) return;
    const minIntervalMs = Math.max(16, Math.round(1000 / Math.max(maxFps, 1)));
    const delay = Math.max(0, minIntervalMs - (Date.now() - lastFlushAtRef.current));
    flushTimerRef.current = setTimeout(flushPending, delay);
  }, [flushPending, maxFps]);

  const connect = useCallback(() => {
    if (!url || !enabled || unmountedRef.current) return;

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
        pendingDeviceIdsRef.current = new Set(devices);
        scheduleFlush();
        const nextDeviceIds =
          subscribedDeviceIdsRef.current.length > 0
            ? subscribedDeviceIdsRef.current
            : devices;
        subscribeToDevices(ws, nextDeviceIds);
        return;
      }

      if (m.type === "frame_update") {
        const frame = normalizeFrameUpdate(m);
        pendingFramesRef.current[frame.device_id] = frame;
        pendingDeviceIdsRef.current.add(frame.device_id);
        scheduleFlush();
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
  }, [enabled, scheduleFlush, url]);

  useEffect(() => {
    subscribedDeviceIdsRef.current = subscribedDeviceIds;
  }, [subscribedDeviceIds]);

  useEffect(() => {
    unmountedRef.current = false;
    if (!enabled) {
      setConnectionState("disconnected");
      return () => {
        unmountedRef.current = true;
      };
    }

    connect();
    return () => {
      unmountedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
      const ws = wsRef.current;
      if (ws) {
        ws.onclose = null; // prevent onclose from scheduling a reconnect on intentional close
        ws.close();
      }
    };
  }, [connect, enabled]);

  useEffect(() => {
    const nextDeviceIds =
      subscribedDeviceIds.length > 0
        ? subscribedDeviceIds
        : deviceIds;
    subscribeToDevices(wsRef.current, nextDeviceIds);
  }, [subscribedDeviceIds, deviceIds]);

  return { connectionState, deviceIds, framesByDevice };
}
