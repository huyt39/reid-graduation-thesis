import BaseApiClient, { type ApiResponse } from "./base-client";
import type { Device } from "@/types";

class DevicesClient extends BaseApiClient {
  list(): Promise<ApiResponse<{ devices: Device[] }>> {
    return this.get<{ devices: Device[] }>("/devices");
  }

  getOne(id: string): Promise<ApiResponse<Device>> {
    return this.get<Device>(`/devices/${id}`);
  }
}

export const devicesClient = new DevicesClient();
