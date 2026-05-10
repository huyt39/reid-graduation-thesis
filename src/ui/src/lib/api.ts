import { clearToken, ensureDevToken, getToken } from "./auth";
import type {
  Device,
  NLQueryResult,
  PaginatedPersons,
  PaginatedSightings,
  PaginatedTimeline,
  Person,
  SimilarPersonsResponse,
  Stats,
} from "@/types";

const BASE = () => process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:18080";

export async function apiFetch<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const requestPath = `${BASE()}/api/v1/${path.replace(/^\//, "")}`;

  async function performRequest(retryWithFreshToken = false): Promise<Response> {
    let token = getToken();
    if (!token || retryWithFreshToken) {
      clearToken();
      token = await ensureDevToken();
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(init.headers as Record<string, string>),
    };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    return fetch(requestPath, {
      ...init,
      headers,
    });
  }

  let res = await performRequest(false);
  if (res.status === 401) {
    res = await performRequest(true);
  }

  if (res.status === 401) {
    clearToken();
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }

  return res.json() as Promise<T>;
}

// ─── Typed helpers ───────────────────────────────────────────────────

function qs(params: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  }
  return parts.length ? `?${parts.join("&")}` : "";
}

// /api/v1/persons accepts: gender, device, is_active, page, page_size as query params.
export function listPersons(
  opts: {
    gender?: string;
    device?: string;
    is_active?: boolean;
    page?: number;
    page_size?: number;
  } = {},
): Promise<PaginatedPersons> {
  const { page = 1, page_size = 20, ...rest } = opts;
  return apiFetch<PaginatedPersons>(
    `persons${qs({ ...rest, page, page_size })}`,
  );
}

export function getPerson(id: number): Promise<Person> {
  return apiFetch<Person>(`persons/${id}`);
}

export function getPersonTimeline(
  id: number,
  options: { start_time?: string; end_time?: string; page?: number; page_size?: number } = {},
): Promise<PaginatedTimeline> {
  return apiFetch<PaginatedTimeline>(`persons/${id}/timeline${qs(options)}`);
}

export function getPersonSightings(
  id: number,
  page = 1,
  pageSize = 20,
): Promise<PaginatedSightings> {
  return apiFetch<PaginatedSightings>(
    `persons/${id}/sightings${qs({ page, page_size: pageSize })}`,
  );
}

export function getSimilarPersons(
  id: number,
  topK = 10,
): Promise<SimilarPersonsResponse> {
  return apiFetch<SimilarPersonsResponse>(
    `persons/${id}/similar${qs({ top_k: topK })}`,
  );
}

export function getDevices(): Promise<{ devices: Device[] }> {
  return apiFetch<{ devices: Device[] }>("devices");
}

export function getStats(): Promise<Stats> {
  return apiFetch<Stats>("stats");
}

export function naturalQuery(query: string): Promise<NLQueryResult> {
  return apiFetch<NLQueryResult>("query/natural", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function structuredSearch(
  queryType: string,
  params: Record<string, unknown>,
): Promise<unknown> {
  return apiFetch<unknown>("search", {
    method: "POST",
    body: JSON.stringify({ query_type: queryType, params }),
  });
}
