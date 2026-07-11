"""MinIO object storage for person/tracklet snapshot images."""
from __future__ import annotations

import io

import structlog

log = structlog.get_logger()

_BUCKET = "reid-snapshots"


class MinIOSnapshotStore:
    # kết nối minio và chuẩn bị bucket lưu snapshot
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minio",
        secret_key: str = "minio123",
        secure: bool = False,
    ) -> None:
        from minio import Minio
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._available = False
        try:
            self._ensure_bucket()
            self._available = True
        except Exception as err:
            log.error("minio.init_failed", error=str(err), exc_info=True)

    # tạo bucket snapshot nếu chưa tồn tại
    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(_BUCKET):
            self._client.make_bucket(_BUCKET)
            log.info("minio.bucket_created", bucket=_BUCKET)

    # upload bytes ảnh jpeg vào minio theo object key
    def upload_snapshot(self, key: str, image_bytes: bytes) -> str:
        """Upload JPEG bytes and return the object key."""
        if not self._available:
            return "    "
        try:
            self._client.put_object(
                _BUCKET,
                key,
                io.BytesIO(image_bytes),
                length=len(image_bytes),
                content_type="image/jpeg",
            )
            return key
        except Exception as err:
            log.error("minio.upload_failed", key=key, error=str(err), exc_info=True)
            return ""

    # lưu ảnh đại diện tốt nhất của một person
    def upload_person_snapshot(self, person_id: int, image_bytes: bytes) -> str:
        return self.upload_snapshot(f"persons/{person_id}/best.jpg", image_bytes)

    # lưu ảnh tốt nhất của một tracklet
    def upload_tracklet_snapshot(self, tracklet_id: str, image_bytes: bytes) -> str:
        return self.upload_snapshot(f"tracklets/{tracklet_id}/best.jpg", image_bytes)

    # lưu snapshot theo từng frame của tracklet
    def upload_tracklet_frame_snapshot(
        self,
        tracklet_id: str,
        frame_idx: int,
        image_bytes: bytes,
    ) -> str:
        return self.upload_snapshot(f"tracklets/{tracklet_id}/frames/{frame_idx}.jpg", image_bytes)

    # tạo url tạm thời để xem hoặc tải ảnh snapshot
    def presigned_url(self, key: str, expires_hours: int = 1) -> str:
        from datetime import timedelta
        if not key or not self._available:
            return ""
        try:
            return self._client.presigned_get_object(_BUCKET, key, expires=timedelta(hours=expires_hours))
        except Exception as err:
            log.error("minio.presigned_failed", key=key, error=str(err), exc_info=True)
            return ""
