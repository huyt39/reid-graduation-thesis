import BaseApiClient, { type ApiResponse } from "./base-client";

interface PresignResponse {
  url: string;
  expires_at?: string;
}

class SnapshotsClient extends BaseApiClient {
  presign(key: string, expiresHours = 1): Promise<ApiResponse<PresignResponse>> {
    return this.get<PresignResponse>("/snapshots/presign", {
      key,
      expires_hours: expiresHours,
    });
  }
}

export const snapshotsClient = new SnapshotsClient();
