import BaseApiClient, { type ApiResponse } from "./base-client";
import type {
  Person,
  PaginatedPersons,
  PaginatedSightings,
  PaginatedTimeline,
  SimilarPersonsResponse,
} from "@/types";

export interface PersonsListParams {
  gender?: string;
  device?: string;
  is_active?: boolean;
  page?: number;
  page_size?: number;
}

class PersonsClient extends BaseApiClient {
  list(params: PersonsListParams = {}): Promise<ApiResponse<PaginatedPersons>> {
    const { page = 1, page_size = 20, ...rest } = params;
    return this.get<PaginatedPersons>("/persons", { ...rest, page, page_size });
  }

  getById(id: number): Promise<ApiResponse<Person>> {
    return super.get<Person>(`/persons/${id}`);
  }

  sightings(
    id: number,
    params: { start_time?: string; end_time?: string; page?: number; page_size?: number } = {}
  ): Promise<ApiResponse<PaginatedSightings>> {
    const { page = 1, page_size = 20, ...rest } = params;
    return super.get<PaginatedSightings>(`/persons/${id}/sightings`, {
      ...rest,
      page,
      page_size,
    });
  }

  timeline(
    id: number,
    params: { start_time?: string; end_time?: string; page?: number; page_size?: number } = {}
  ): Promise<ApiResponse<PaginatedTimeline>> {
    const { page = 1, page_size = 50, ...rest } = params;
    return super.get<PaginatedTimeline>(`/persons/${id}/timeline`, {
      ...rest,
      page,
      page_size,
    });
  }

  similar(
    id: number,
    params: { top_k?: number; min_score?: number } = {}
  ): Promise<ApiResponse<SimilarPersonsResponse>> {
    const { top_k = 10, min_score } = params;
    return super.get<SimilarPersonsResponse>(`/persons/${id}/similar`, {
      top_k,
      min_score,
    });
  }
}

export const personsClient = new PersonsClient();
