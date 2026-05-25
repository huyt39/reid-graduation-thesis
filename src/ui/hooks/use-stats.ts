"use client";

import useSWR from "swr";
import { statsClient } from "@/lib/api/stats-client";
import type { Stats } from "@/types";

export const STATS_CACHE_KEY = "stats:overview";

const fetcher = async (): Promise<Stats> => {
  const response = await statsClient.getStats();
  if (response.error || !response.data) {
    throw new Error(response.error || "Failed to load stats");
  }
  return response.data;
};

export function useStats() {
  return useSWR<Stats>(STATS_CACHE_KEY, fetcher, {
    revalidateOnFocus: true,
    revalidateOnReconnect: true,
    dedupingInterval: 30000,
    keepPreviousData: true,
  });
}
