// Mirror of src/query_service/src/schemas/query.py response models.
// Keep field names and shapes in sync — the gateway proxies these unchanged.

export interface PersonAttributes {
  [key: string]: number | string | undefined;
  gender: string;
  gender_confidence: number;
  age_child?: string;
  age_child_confidence?: number;
  backpack?: string;
  backpack_confidence?: number;
  sidebag?: string;
  sidebag_confidence?: number;
  hat?: string;
  hat_confidence?: number;
  glasses?: string;
  glasses_confidence?: number;
  sleeve?: string;
  sleeve_confidence?: number;
  lower?: string;
  lower_confidence?: number;
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

export interface TrackletQuality {
  v_avg: number;
  embedding_consistency: number;
  bbox_size_stability: number;
  position_stability: number;
  good_frame_ratio: number;
  overall_consistency: number;
}

export interface TrackletMatching {
  method: string;
  source: string;
  similarity_score: number | null;
  runner_up_score: number | null;
  margin_to_runner_up: number | null;
  reuse_person_id: number | null;
  tentative_attempts: number | null;
  canonical_update_applied: boolean | null;
}

export interface TrackletFrameSample {
  frame_idx: number;
  visibility_score: number;
  overlap_ratio: number;
  selected: boolean;
  selection_reason: string;
  crop_url: string | null;
}

export interface TrackletEvidence {
  selected_frame_count: number;
  selected_frame_indices: number[];
  frame_samples: TrackletFrameSample[];
}

export interface Tracklet {
  tracklet_id: string;
  track_id: number;
  person_id: number | null;
  device_id: string;
  state: string;
  frame_range: Record<string, number>;
  entry_count: number;
  quality: TrackletQuality;
  matching: TrackletMatching;
  evidence: TrackletEvidence;
  best_crop_url: string | null;
  created_at: string | null;
}

export interface OcclusionCandidate {
  candidate_id: string;
  track_id: number;
  device_id: string;
  reason: string;
  status: string;
  frame_range: Record<string, number>;
  entry_count: number;
  quality: TrackletQuality;
  matching: TrackletMatching;
  evidence: TrackletEvidence;
  best_crop_url: string | null;
  created_at: string | null;
  updated_at: string | null;
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

export interface PaginatedTracklets {
  items: Tracklet[];
  total: number;
  page: number;
  page_size: number;
}

export interface PaginatedOcclusionCandidates {
  items: OcclusionCandidate[];
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
  summary?: string;
  message?: string;
}

export interface PersonSearchFilters {
  person_id?: number;
  gender?: string;
  gender_confidence_min?: number;
  age_child?: string;
  backpack?: string;
  sidebag?: string;
  hat?: string;
  glasses?: string;
  sleeve?: string;
  lower?: string;
  last_seen_device?: string;
  first_seen_after?: string;
  first_seen_before?: string;
  last_seen_after?: string;
  last_seen_before?: string;
  min_sighting_count?: number;
  is_active?: boolean;
}
