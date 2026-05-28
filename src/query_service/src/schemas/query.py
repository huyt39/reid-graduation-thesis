"""Pydantic models for query service requests and responses."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# Requests

class NLQueryRequest(BaseModel):
    query: str


class StructuredQueryRequest(BaseModel):
    query_type: str
    params: Any = Field(default_factory=dict)


class PersonLookupParams(BaseModel):
    person_id: int


class DeviceLookupParams(BaseModel):
    device_id: str | None = None


class TimelineParams(BaseModel):
    person_id: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    event_types: list[str] | None = None


class SimilaritySearchParams(BaseModel):
    person_id: int
    top_k: int = 10
    min_score: float = 0.5


class SightingAggregationParams(BaseModel):
    person_id: int | None = None
    device_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    group_by: Literal["hour", "day", "device"] = "hour"


class PersonSearchFilters(BaseModel):
    person_id: int | None = None
    gender: str | None = None
    gender_confidence_min: float | None = None
    age_child: str | None = None
    backpack: str | None = None
    sidebag: str | None = None
    hat: str | None = None
    glasses: str | None = None
    sleeve: str | None = None
    lower: str | None = None
    last_seen_device: str | None = None
    first_seen_after: datetime | None = None
    first_seen_before: datetime | None = None
    last_seen_after: datetime | None = None
    last_seen_before: datetime | None = None
    min_sighting_count: int | None = None
    is_active: bool | None = None


class PersonSearchParams(BaseModel):
    filters: PersonSearchFilters = Field(default_factory=PersonSearchFilters)
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class PersonLookupQuery(StructuredQueryRequest):
    query_type: Literal["person_lookup"]
    params: PersonLookupParams


class PersonSearchQuery(StructuredQueryRequest):
    query_type: Literal["person_search"]
    params: PersonSearchParams


class TimelineQuery(StructuredQueryRequest):
    query_type: Literal["timeline"]
    params: TimelineParams


class SimilaritySearchQuery(StructuredQueryRequest):
    query_type: Literal["similarity_search"]
    params: SimilaritySearchParams


class SightingAggregationQuery(StructuredQueryRequest):
    query_type: Literal["sighting_aggregation"]
    params: SightingAggregationParams


class DeviceLookupQuery(StructuredQueryRequest):
    query_type: Literal["device_lookup"]
    params: DeviceLookupParams


StructuredSearchQuery = Annotated[
    PersonLookupQuery
    | PersonSearchQuery
    | TimelineQuery
    | SimilaritySearchQuery
    | SightingAggregationQuery
    | DeviceLookupQuery,
    Field(discriminator="query_type"),
]


class Pagination(BaseModel):
    page: int = 1
    page_size: int = Field(20, le=100)


class SortSpec(BaseModel):
    field: str = "stats.last_seen_at"
    order: str = "desc"  # "asc" or "desc"


# Responses

class PersonAttributes(BaseModel):
    model_config = ConfigDict(extra="allow")

    gender: str = "unknown"
    gender_confidence: float = 0.0
    age_child: str = "unknown"
    age_child_confidence: float = 0.0
    backpack: str = "unknown"
    backpack_confidence: float = 0.0
    sidebag: str = "unknown"
    sidebag_confidence: float = 0.0
    hat: str = "unknown"
    hat_confidence: float = 0.0
    glasses: str = "unknown"
    glasses_confidence: float = 0.0
    sleeve: str = "unknown"
    sleeve_confidence: float = 0.0
    lower: str = "unknown"
    lower_confidence: float = 0.0


class PersonStats(BaseModel):
    sighting_count: int = 0
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_seen_device: str = ""


class PersonResponse(BaseModel):
    person_id: int
    attributes: PersonAttributes = Field(default_factory=PersonAttributes)
    stats: PersonStats = Field(default_factory=PersonStats)
    snapshot_url: str | None = None
    source: str = ""
    is_active: bool = True


class SightingResponse(BaseModel):
    person_id: int
    device_id: str
    tracklet_id: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float = 0.0
    quality_score: float = 0.0
    snapshot_url: str | None = None
    attributes: PersonAttributes = Field(default_factory=PersonAttributes)


class TrackletQuality(BaseModel):
    v_avg: float = 0.0
    embedding_consistency: float = 0.0
    bbox_size_stability: float = 0.0
    position_stability: float = 0.0
    good_frame_ratio: float = 0.0
    overall_consistency: float = 0.0


class TrackletMatching(BaseModel):
    method: str = ""
    source: str = ""
    similarity_score: float | None = None
    runner_up_score: float | None = None
    margin_to_runner_up: float | None = None
    reuse_person_id: int | None = None
    tentative_attempts: int | None = None
    canonical_update_applied: bool | None = None


class TrackletFrameSample(BaseModel):
    frame_idx: int
    visibility_score: float = 0.0
    overlap_ratio: float = 0.0
    selected: bool = False
    selection_reason: str = ""
    crop_url: str | None = None


class TrackletEvidence(BaseModel):
    selected_frame_count: int = 0
    selected_frame_indices: list[int] = Field(default_factory=list)
    frame_samples: list[TrackletFrameSample] = Field(default_factory=list)


class TrackletResponse(BaseModel):
    tracklet_id: str
    track_id: int
    person_id: int | None = None
    device_id: str
    state: str
    frame_range: dict[str, int] = Field(default_factory=dict)
    entry_count: int = 0
    quality: TrackletQuality = Field(default_factory=TrackletQuality)
    matching: TrackletMatching = Field(default_factory=TrackletMatching)
    evidence: TrackletEvidence = Field(default_factory=TrackletEvidence)
    best_crop_url: str | None = None
    created_at: datetime | None = None


class OcclusionCandidateResponse(BaseModel):
    candidate_id: str
    track_id: int
    device_id: str
    reason: str
    status: str = "unconfirmed"
    frame_range: dict[str, int] = Field(default_factory=dict)
    entry_count: int = 0
    quality: TrackletQuality = Field(default_factory=TrackletQuality)
    matching: TrackletMatching = Field(default_factory=TrackletMatching)
    evidence: TrackletEvidence = Field(default_factory=TrackletEvidence)
    best_crop_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TimelineEvent(BaseModel):
    person_id: int
    event_type: str
    timestamp: datetime
    device_id: str = ""
    details: dict[str, object] = Field(default_factory=dict)


class DeviceResponse(BaseModel):
    device_id: str
    name: str = ""
    location: str = ""
    status: str = "unknown"
    last_frame_at: datetime | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    sighting_count: int = 0
    unique_person_count: int = 0


class StatsResponse(BaseModel):
    total_persons: int = 0
    active_persons: int = 0
    total_sightings: int = 0
    total_devices: int = 0


class SimilarPersonResult(BaseModel):
    person_id: int
    score: float
    attributes: PersonAttributes = Field(default_factory=PersonAttributes)


class PaginatedResponse(BaseModel):
    items: list[dict] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class PaginatedPersonsResponse(BaseModel):
    items: list[PersonResponse] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class PaginatedSightingsResponse(BaseModel):
    items: list[SightingResponse] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class PaginatedTrackletsResponse(BaseModel):
    items: list[TrackletResponse] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class PaginatedOcclusionCandidatesResponse(BaseModel):
    items: list[OcclusionCandidateResponse] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class PaginatedTimelineResponse(BaseModel):
    items: list[TimelineEvent] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class SimilarPersonItem(BaseModel):
    person_id: int
    score: float
    person: PersonResponse | None = None


class SimilarPersonsResponse(BaseModel):
    similar_persons: list[SimilarPersonItem] = Field(default_factory=list)


class AggregationResponse(BaseModel):
    aggregation: list[dict] = Field(default_factory=list)


class DevicesListResponse(BaseModel):
    devices: list[DeviceResponse] = Field(default_factory=list)
