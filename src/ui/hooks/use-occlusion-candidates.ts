"use client";

import useSWR from "swr";
import {
  occlusionClient,
  type OcclusionCandidatesParams,
} from "@/lib/api/occlusion-client";
import type { PaginatedOcclusionCandidates } from "@/types";

const fetcher = async (
  _key: string,
  params: OcclusionCandidatesParams
): Promise<PaginatedOcclusionCandidates> => {
  const response = await occlusionClient.list(params);
  if (response.error || !response.data) {
    throw new Error(response.error || "Failed to load occlusion candidates");
  }
  return response.data;
};

export function useOcclusionCandidates(params: OcclusionCandidatesParams = {}) {
  return useSWR<PaginatedOcclusionCandidates>(
    ["occlusion-candidates:list", params],
    ([, p]) => fetcher("occlusion-candidates:list", p as OcclusionCandidatesParams),
    {
      revalidateOnFocus: true,
      dedupingInterval: 10000,
      keepPreviousData: true,
    }
  );
}
