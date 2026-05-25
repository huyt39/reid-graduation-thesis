"use client";

import useSWR from "swr";
import { statsClient } from "@/lib/api/stats-client";

export interface AggregateBucket {
  _id: string;
  count: number;
  total_duration: number;
  avg_quality: number;
}

type AggregateParams = Parameters<typeof statsClient.aggregate>[0];

const fetcher = async (params: AggregateParams): Promise<AggregateBucket[]> => {
  const response = await statsClient.aggregate(params);
  if (response.error || !response.data) {
    throw new Error(response.error || "Failed to load aggregate data");
  }
  return response.data.aggregation as unknown as AggregateBucket[];
};

export function useAggregate(params: AggregateParams = {}) {
  return useSWR<AggregateBucket[]>(
    ["stats:aggregate", params],
    ([, p]) => fetcher(p as AggregateParams),
    {
      revalidateOnFocus: true,
      dedupingInterval: 60000,
      keepPreviousData: true,
    }
  );
}
