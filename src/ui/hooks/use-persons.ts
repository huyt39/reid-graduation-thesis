"use client";

import { useEffect } from "react";
import useSWR from "swr";
import { personsClient, type PersonsListParams } from "@/lib/api/persons-client";
import type { PaginatedPersons } from "@/types";

const fetcher = async (
  _key: string,
  params: PersonsListParams
): Promise<PaginatedPersons> => {
  const response = await personsClient.list(params);
  if (response.error || !response.data) {
    throw new Error(response.error || "Failed to load persons");
  }
  return response.data;
};

type PersonsRealtimeListener = () => void;

const WS_URL = process.env.NEXT_PUBLIC_STREAMING_WS || "ws://localhost:8765";
const PERSONS_SIGNAL_MIN_INTERVAL_MS = 750;
const PERSONS_WS_BACKOFF_CAP_MS = 8000;

let personsRealtimeWs: WebSocket | null = null;
let personsRealtimeReconnectTimer: ReturnType<typeof setTimeout> | null = null;
let personsRealtimeBackoffMs = 1000;
let personsRealtimeLastSignalAt = 0;
const personsRealtimeListeners = new Set<PersonsRealtimeListener>();

function notifyPersonsRealtimeListeners() {
  const now = Date.now();
  if (now - personsRealtimeLastSignalAt < PERSONS_SIGNAL_MIN_INTERVAL_MS) {
    return;
  }
  personsRealtimeLastSignalAt = now;
  for (const listener of personsRealtimeListeners) {
    listener();
  }
}

function schedulePersonsRealtimeReconnect() {
  if (personsRealtimeReconnectTimer || personsRealtimeListeners.size === 0) {
    return;
  }
  personsRealtimeReconnectTimer = setTimeout(() => {
    personsRealtimeReconnectTimer = null;
    ensurePersonsRealtimeConnection();
  }, personsRealtimeBackoffMs);
  personsRealtimeBackoffMs = Math.min(
    personsRealtimeBackoffMs * 2,
    PERSONS_WS_BACKOFF_CAP_MS
  );
}

function closePersonsRealtimeConnection() {
  if (personsRealtimeReconnectTimer) {
    clearTimeout(personsRealtimeReconnectTimer);
    personsRealtimeReconnectTimer = null;
  }
  if (personsRealtimeWs) {
    const ws = personsRealtimeWs;
    personsRealtimeWs = null;
    ws.onopen = null;
    ws.onmessage = null;
    ws.onerror = null;
    ws.onclose = null;
    ws.close();
  }
}

function ensurePersonsRealtimeConnection() {
  if (typeof window === "undefined" || personsRealtimeWs || personsRealtimeListeners.size === 0) {
    return;
  }

  const ws = new WebSocket(`${WS_URL}/ws`);
  personsRealtimeWs = ws;

  ws.onopen = () => {
    personsRealtimeBackoffMs = 1000;
  };

  ws.onmessage = (event) => {
    let message: unknown;
    try {
      message = JSON.parse(event.data as string);
    } catch {
      return;
    }

    const payload = message as {
      type?: unknown;
      tracked_persons?: unknown;
    };
    if (payload.type !== "frame_update" || !Array.isArray(payload.tracked_persons)) {
      return;
    }

    const hasResolvedPerson = payload.tracked_persons.some((person) => {
      const personId = (person as { person_id?: unknown })?.person_id;
      return typeof personId === "number" && Number.isFinite(personId);
    });
    if (hasResolvedPerson) {
      notifyPersonsRealtimeListeners();
    }
  };

  ws.onerror = () => {
    ws.close();
  };

  ws.onclose = () => {
    if (personsRealtimeWs === ws) {
      personsRealtimeWs = null;
    }
    schedulePersonsRealtimeReconnect();
  };
}

function subscribeToPersonsRealtime(listener: PersonsRealtimeListener) {
  personsRealtimeListeners.add(listener);
  ensurePersonsRealtimeConnection();

  return () => {
    personsRealtimeListeners.delete(listener);
    if (personsRealtimeListeners.size === 0) {
      closePersonsRealtimeConnection();
    }
  };
}

export function usePersons(params: PersonsListParams = {}) {
  const refreshInterval = params.is_active === true ? 1000 : 1500;
  const dedupingInterval = Math.max(500, refreshInterval - 100);

  const swr = useSWR<PaginatedPersons>(
    ["persons:list", params],
    ([, p]) => fetcher("persons:list", p as PersonsListParams),
    {
      revalidateOnFocus: true,
      revalidateOnReconnect: true,
      refreshInterval,
      dedupingInterval,
      keepPreviousData: true,
    }
  );
  const { mutate } = swr;

  useEffect(() => {
    let lastRevalidateAt = 0;
    return subscribeToPersonsRealtime(() => {
      const now = Date.now();
      if (now - lastRevalidateAt < dedupingInterval) {
        return;
      }
      lastRevalidateAt = now;
      void mutate();
    });
  }, [dedupingInterval, mutate]);

  return swr;
}
