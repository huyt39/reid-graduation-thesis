from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from src.api import deps
from src.api.routes import persons as persons_routes


@pytest.mark.asyncio
async def test_get_person_uses_cache(monkeypatch):
    mongo = AsyncMock()
    redis_cache = AsyncMock()
    redis_cache.get_person.return_value = {"person_id": 7}
    minio_urls = MagicMock()

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(persons_routes, "get_redis", lambda: redis_cache)
    monkeypatch.setattr(persons_routes, "get_minio_urls", lambda: minio_urls)

    result = await persons_routes.get_person(7)

    assert result["person_id"] == 7
    mongo.get_person.assert_not_awaited()
    redis_cache.set_person.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_person_attaches_snapshot_url_and_caches_enriched_person(monkeypatch):
    mongo = AsyncMock()
    redis_cache = AsyncMock()
    minio_urls = MagicMock()
    redis_cache.get_person.return_value = None
    mongo.get_person.return_value = {"person_id": 7, "snapshot_key": "persons/7/best.jpg"}
    minio_urls.presigned_url.return_value = "https://example.com/p/7.jpg"

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(persons_routes, "get_redis", lambda: redis_cache)
    monkeypatch.setattr(persons_routes, "get_minio_urls", lambda: minio_urls)

    result = await persons_routes.get_person(7)

    assert result["snapshot_url"] == "https://example.com/p/7.jpg"
    redis_cache.set_person.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_persons_attaches_snapshot_urls(monkeypatch):
    mongo = AsyncMock()
    minio_urls = MagicMock()
    mongo.search_persons.return_value = (
        [
            {"person_id": 1, "snapshot_key": "persons/1/best.jpg"},
            {"person_id": 2, "snapshot_key": None},
        ],
        2,
    )
    minio_urls.presigned_url.side_effect = [
        "https://example.com/p/1.jpg",
        None,
    ]

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(persons_routes, "get_minio_urls", lambda: minio_urls)

    result = await persons_routes.list_persons(page=1, page_size=20)

    assert result["items"][0]["snapshot_url"] == "https://example.com/p/1.jpg"
    assert result["items"][1]["snapshot_url"] is None
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_get_sightings_attaches_snapshot_urls(monkeypatch):
    mongo = AsyncMock()
    minio_urls = MagicMock()
    mongo.get_sightings.return_value = (
        [
            {"person_id": 7, "device_id": "cam-1", "tracklet_id": "t1", "started_at": datetime(2026, 1, 1, 0, 0, 0),
             "ended_at": datetime(2026, 1, 1, 0, 0, 10), "snapshot_key": "sightings/1.jpg"},
            {"person_id": 7, "device_id": "cam-2", "tracklet_id": "t2", "started_at": datetime(2026, 1, 1, 0, 1, 0),
             "ended_at": datetime(2026, 1, 1, 0, 1, 10), "snapshot_key": None},
        ],
        2,
    )
    minio_urls.presigned_url.side_effect = [
        "https://example.com/s/1.jpg",
        None,
    ]

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(persons_routes, "get_minio_urls", lambda: minio_urls)

    result = await persons_routes.get_sightings(person_id=7, page=1, page_size=20)

    assert result["items"][0]["snapshot_url"] == "https://example.com/s/1.jpg"
    assert result["items"][1]["snapshot_url"] is None
    assert result["total"] == 2


@pytest.mark.asyncio
async def test_get_similar_enriches_persons(monkeypatch):
    mongo = AsyncMock()
    minio_urls = MagicMock()
    mongo.get_person = AsyncMock(side_effect=[
        {"person_id": 1},
        {"person_id": 2, "attributes": {"gender": "male"}, "snapshot_key": "persons/2/best.jpg"},
    ])
    qdrant = MagicMock()
    qdrant.search_similar.return_value = [{"person_id": 2, "score": 0.91}]
    minio_urls.presigned_url.return_value = "https://example.com/p/2.jpg"

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(persons_routes, "get_minio_urls", lambda: minio_urls)
    monkeypatch.setattr(deps, "get_qdrant", lambda: qdrant)

    result = await persons_routes.get_similar(person_id=1, top_k=5, min_score=0.5)

    assert result["similar_persons"][0]["person_id"] == 2
    assert result["similar_persons"][0]["person"]["person_id"] == 2
    assert result["similar_persons"][0]["person"]["snapshot_url"] == "https://example.com/p/2.jpg"


@pytest.mark.asyncio
async def test_get_similar_raises_404_for_missing_source_person(monkeypatch):
    mongo = AsyncMock()
    mongo.get_person = AsyncMock(return_value=None)
    qdrant = MagicMock()
    minio_urls = MagicMock()

    monkeypatch.setattr(persons_routes, "get_mongo", lambda: mongo)
    monkeypatch.setattr(persons_routes, "get_minio_urls", lambda: minio_urls)
    monkeypatch.setattr(deps, "get_qdrant", lambda: qdrant)

    with pytest.raises(HTTPException) as exc:
        await persons_routes.get_similar(person_id=999, top_k=5, min_score=0.5)

    assert exc.value.status_code == 404
