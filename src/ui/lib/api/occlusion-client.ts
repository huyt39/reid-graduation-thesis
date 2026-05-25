import BaseApiClient, { type ApiResponse } from "./base-client";
import type { PaginatedOcclusionCandidates } from "@/types";

export interface OcclusionCandidatesParams {
  status?: string | null;
  device?: string;
  page?: number;
  page_size?: number;
}

class OcclusionClient extends BaseApiClient {
  list(
    params: OcclusionCandidatesParams = {}
  ): Promise<ApiResponse<PaginatedOcclusionCandidates>> {
    const { page = 1, page_size = 20, ...rest } = params;
    return this.get<PaginatedOcclusionCandidates>("/occlusion-candidates", {
      ...rest,
      page,
      page_size,
    });
  }
}

export const occlusionClient = new OcclusionClient();
