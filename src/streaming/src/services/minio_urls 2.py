from __future__ import annotations

from datetime import timedelta


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
        public = public_endpoint or internal_endpoint

        self._internal_client = Minio(
            internal_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._public_client = Minio(
            public,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region="us-east-1",
        )

    def presigned_url(self, object_key: str | None, expires_hours: int = 1) -> str | None:
        if not object_key:
            return None

        return self._public_client.presigned_get_object(
            self._bucket,
            object_key,
            expires=timedelta(hours=expires_hours),
        )

    def ping(self) -> bool:
        try:
            self._internal_client.bucket_exists(self._bucket)
            return True
        except Exception:
            return False
