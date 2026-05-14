"use client";

import useSWR from "swr";
import { personsClient } from "@/lib/api/persons-client";
import type {
  PaginatedSightings,
  PaginatedTracklets,
  PaginatedTimeline,
  Person,
  SimilarPersonsResponse,
} from "@/types";

export function usePerson(id: number | null) {
  return useSWR<Person>(
    id ? ["person", id] : null,
    async ([, personId]) => {
      const response = await personsClient.getById(personId as number);
      if (response.error || !response.data) {
        throw new Error(response.error || "Failed to load person");
      }
      return response.data;
    },
    { revalidateOnFocus: true, dedupingInterval: 30000 }
  );
}

export function usePersonSightings(id: number | null, page = 1, pageSize = 20) {
  return useSWR<PaginatedSightings>(
    id ? ["person:sightings", id, page, pageSize] : null,
    async ([, personId, p, ps]) => {
      const response = await personsClient.sightings(personId as number, {
        page: p as number,
        page_size: ps as number,
      });
      if (response.error || !response.data) {
        throw new Error(response.error || "Failed to load sightings");
      }
      return response.data;
    },
    { keepPreviousData: true }
  );
}

export function usePersonTimeline(id: number | null, page = 1, pageSize = 50) {
  return useSWR<PaginatedTimeline>(
    id ? ["person:timeline", id, page, pageSize] : null,
    async ([, personId, p, ps]) => {
      const response = await personsClient.timeline(personId as number, {
        page: p as number,
        page_size: ps as number,
      });
      if (response.error || !response.data) {
        throw new Error(response.error || "Failed to load timeline");
      }
      return response.data;
    },
    { keepPreviousData: true }
  );
}

export function usePersonTracklets(id: number | null, page = 1, pageSize = 20) {
  return useSWR<PaginatedTracklets>(
    id ? ["person:tracklets", id, page, pageSize] : null,
    async ([, personId, p, ps]) => {
      const response = await personsClient.tracklets(personId as number, {
        page: p as number,
        page_size: ps as number,
      });
      if (response.error || !response.data) {
        throw new Error(response.error || "Failed to load tracklets");
      }
      return response.data;
    },
    { keepPreviousData: true }
  );
}

export function usePersonSimilar(id: number | null, topK = 10) {
  return useSWR<SimilarPersonsResponse>(
    id ? ["person:similar", id, topK] : null,
    async ([, personId, k]) => {
      const response = await personsClient.similar(personId as number, {
        top_k: k as number,
      });
      if (response.error || !response.data) {
        throw new Error(response.error || "Failed to load similar persons");
      }
      return response.data;
    },
    { dedupingInterval: 60000 }
  );
}
