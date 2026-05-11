"use client";

import useSWR from "swr";
import { devicesClient } from "@/lib/api/devices-client";
import type { Device } from "@/types";

export function useDevices() {
  return useSWR<{ devices: Device[] }>(
    "devices:list",
    async () => {
      const response = await devicesClient.list();
      if (response.error || !response.data) {
        throw new Error(response.error || "Failed to load devices");
      }
      return response.data;
    },
    { revalidateOnFocus: true, dedupingInterval: 30000 }
  );
}

export function useDevice(id: string | null) {
  return useSWR<Device>(
    id ? ["device", id] : null,
    async ([, deviceId]) => {
      const response = await devicesClient.getOne(deviceId as string);
      if (response.error || !response.data) {
        throw new Error(response.error || "Failed to load device");
      }
      return response.data;
    },
    { dedupingInterval: 30000 }
  );
}
