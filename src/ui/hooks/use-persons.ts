"use client";

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

export function usePersons(params: PersonsListParams = {}) {
  return useSWR<PaginatedPersons>(
    ["persons:list", params],
    ([, p]) => fetcher("persons:list", p as PersonsListParams),
    {
      revalidateOnFocus: true,
      dedupingInterval: 15000,
      keepPreviousData: true,
    }
  );
}
