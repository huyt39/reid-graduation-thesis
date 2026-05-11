// Mirror of src/query_service/src/schemas/query.py response models.
// Keep field names and shapes in sync — the gateway proxies these unchanged.

export interface PersonAttributes {
  gender: string;
  gender_confidence: number;
}

export interface PersonStats {
  sighting_count: number;
  first_seen_at: string | null;
  last_seen_at: string | null;
  last_seen_device: string;
}

export interface Person {
  person_id: number;
  attributes: PersonAttributes;
  stats: PersonStats;
  snapshot_url: string | null;
  source: string;
  is_active: boolean;
}

export interface Sighting {
  person_id: number;
  device_id: string;
  tracklet_id: string;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  quality_score: number;
  snapshot_url: string | null;
  attributes: PersonAttributes;
}

export interface TimelineEvent {
  person_id: number;
  event_type: string;
  timestamp: string;
  device_id: string;
  details: Record<string, unknown>;
}

export interface Device {
  device_id: string;
  name: string;
  location: string;
  status: string;
  last_frame_at: string | null;
  first_seen_at: string | null;
  last_seen_at: string | null;
  sighting_count: number;
  unique_person_count: number;
}

export interface Stats {
  total_persons: number;
  active_persons: number;
  total_sightings: number;
  total_devices: number;
}

export interface SimilarPersonItem {
  person_id: number;
  score: number;
  person: Person | null;
}

export interface SimilarPersonsResponse {
  similar_persons: SimilarPersonItem[];
}

export interface PaginatedPersons {
  items: Person[];
  total: number;
  page: number;
  page_size: number;
}

export interface PaginatedSightings {
  items: Sighting[];
  total: number;
  page: number;
  page_size: number;
}

export interface PaginatedTimeline {
  items: TimelineEvent[];
  total: number;
  page: number;
  page_size: number;
}

export interface AggregationBucket {
  [key: string]: unknown;
  count?: number;
}

export interface AggregationResponse {
  aggregation: AggregationBucket[];
}

export interface ParsedQuery {
  query_type: string;
  params: Record<string, unknown>;
}

export interface NLQueryResult {
  parsed_query?: ParsedQuery;
  result?: unknown;
  message?: string;
}

export interface PersonSearchFilters {
  person_id?: number;
  gender?: string;
  gender_confidence_min?: number;
  last_seen_device?: string;
  first_seen_after?: string;
  first_seen_before?: string;
  last_seen_after?: string;
  last_seen_before?: string;
  min_sighting_count?: number;
  is_active?: boolean;
}
