import type { Device } from "@/types";

export function getDeviceDisplayName(device: Pick<Device, "device_id" | "name">): string {
  const match = device.device_id.match(/(\d+)$/);
  if (match) {
    return `Camera ${match[1]}`;
  }

  return device.name || device.device_id;
}

export function getDeviceDisplayLocation(): string {
  return "B1";
}
