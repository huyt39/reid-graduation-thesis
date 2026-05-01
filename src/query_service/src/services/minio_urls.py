from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlsplit, urlunsplit


class MinIOURLBuilder:
    def __init__(
        self,
        internal_endpoint: str,
        public_endpoint: str | None,
        access_key: str,
        secret_key: str,
        bucket: str = "reid-snapshots",
        secure: bool = False,
    ) -> None:
        from minio import Minio

        self._bucket = bucket
        self._secure = secure
        self._public_endpoint = public_endpoint or internal_endpoint

        self._internal_client = Minio(
            internal_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def presigned_url(self, object_key: str | None, expires_hours: int = 1) -> str | None:
        if not object_key:
            return None

        internal_url = self._internal_client.presigned_get_object(
            self._bucket,
            object_key,
            expires=timedelta(hours=expires_hours),
        )

        parts = urlsplit(internal_url)
        scheme = "https" if self._secure else "http"

        return urlunsplit(
            (
                scheme,
                self._public_endpoint,
                parts.path,
                parts.query,
                parts.fragment,
            )
        )

    def ping(self) -> bool:
        try:
            self._internal_client.bucket_exists(self._bucket)
            return True
        except Exception:
            return False
