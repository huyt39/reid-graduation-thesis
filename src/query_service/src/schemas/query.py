"""Pydantic models for query service requests and responses."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from typing import Literal


# Requests 

class NLQueryRequest(BaseModel):
    query: str


class StructuredQueryRequest(BaseModel):
    query_type: Literal[
        "person_lookup",
        "person_search",
        "timeline",
        "similarity_search",
        "sighting_aggregation",
        "device_lookup",
    ]
    params: dict = Field(default_factory=dict)


class PersonSearchFilters(BaseModel):
    gender: str | None = None
    gender_confidence_min: float | None = None
    last_seen_device: str | None = None
    last_seen_after: datetime | None = None
    last_seen_before: datetime | None = None
    is_active: bool | None = None


class Pagination(BaseModel):
    page: int = 1
    page_size: int = Field(20, le=100)


class SortSpec(BaseModel):
    field: str = "stats.last_seen_at"
    order: str = "desc"  # "asc" or "desc"


# Responses

class PersonAttributes(BaseModel):
    gender: str = "unknown"
    gender_confidence: float = 0.0


class PersonStats(BaseModel):
    sighting_count: int = 0
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_seen_device: str = ""


class PersonResponse(BaseModel):
    person_id: int
    attributes: PersonAttributes = PersonAttributes()
    stats: PersonStats = PersonStats()
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
    attributes: PersonAttributes = PersonAttributes()


class TimelineEvent(BaseModel):
    person_id: int
    event_type: str
    timestamp: datetime
    device_id: str = ""
    details: dict = {}


class DeviceResponse(BaseModel):
    device_id: str
    name: str = ""
    location: str = ""
    status: str = "unknown"
    last_frame_at: datetime | None = None


class StatsResponse(BaseModel):
    total_persons: int = 0
    active_persons: int = 0
    total_sightings: int = 0
    total_devices: int = 0


class SimilarPersonResult(BaseModel):
    person_id: int
    score: float
    attributes: PersonAttributes = PersonAttributes()


class PaginatedResponse(BaseModel):
    items: list = []
    total: int = 0
    page: int = 1
    page_size: int = 20
