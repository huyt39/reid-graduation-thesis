from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import get_mongo
from src.schemas.query import DeviceResponse, DevicesListResponse

router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("", response_model=DevicesListResponse)
async def list_devices():
    mongo = get_mongo()
    devices = await mongo.list_devices()
    return {"devices": devices}


@router.get("/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str):
    mongo = get_mongo()
    device = await mongo.get_device(device_id)
    if device is None:
        from fastapi import HTTPException
        raise HTTPException(404, f"Device {device_id} not found")
    return device
