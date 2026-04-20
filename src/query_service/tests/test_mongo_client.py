from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.mongo_client import MongoQueryClient


@pytest.mark.asyncio
async def test_get_stats_falls_back_to_sightings_distinct_for_devices():
    client = MongoQueryClient.__new__(MongoQueryClient)

    db = MagicMock()
    db.persons.count_documents = AsyncMock(side_effect=[5, 3])
    db.sightings.count_documents = AsyncMock(return_value=12)
    db.sightings.distinct = AsyncMock(return_value=["cam_01", "cam_02"])
    db.devices.count_documents = AsyncMock(return_value=0)

    client._db = db

    result = await client.get_stats()

    assert result == {
        "total_persons": 5,
        "active_persons": 3,
        "total_sightings": 12,
        "total_devices": 2,
    }
    db.sightings.distinct.assert_awaited_once_with("device_id")
