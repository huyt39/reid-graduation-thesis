from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api.routes import devices as devices_routes


@pytest.mark.asyncio
async def test_get_device_route_raises_404_for_missing_device(monkeypatch):
    mongo = AsyncMock()
    mongo.get_device.return_value = None

    monkeypatch.setattr(devices_routes, "get_mongo", lambda: mongo)

    with pytest.raises(HTTPException) as exc:
        await devices_routes.get_device("cam_404")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_devices_route_returns_devices_wrapper(monkeypatch):
    mongo = AsyncMock()
    mongo.list_devices.return_value = [
        {
            "device_id": "cam_01",
            "sighting_count": 4,
            "unique_person_count": 2,
        }
    ]

    monkeypatch.setattr(devices_routes, "get_mongo", lambda: mongo)

    result = await devices_routes.list_devices()

    assert result == {
        "devices": [
            {
                "device_id": "cam_01",
                "sighting_count": 4,
                "unique_person_count": 2,
            }
        ]
    }
    mongo.list_devices.assert_awaited_once()


def test_devices_list_response_defaults_to_empty_list():
    from src.schemas.query import DevicesListResponse

    devices = DevicesListResponse()

    assert devices.devices == []
