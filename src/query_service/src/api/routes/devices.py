from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import get_mongo

router = APIRouter(prefix="/devices", tags=["devices"])


@router.get("")
async def list_devices():
    mongo = get_mongo()
    devices = await mongo.list_devices()
    return {"devices": devices}


@router.get("/{device_id}")
async def get_device(device_id: str):
    mongo = get_mongo()
    device = await mongo.get_device(device_id)
    if device is None:
        from fastapi import HTTPException
        raise HTTPException(404, f"Device {device_id} not found")
    return device
