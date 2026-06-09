from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Standalone raw-video MJPEG server. Fully independent of the ReID path:
    it opens its OWN VideoCapture per device and serves MJPET over HTTP, so the
    live UI video is smooth regardless of detection/worker CPU. It does NOT touch
    Kafka, the edge ReID loop or the worker."""

    model_config = SettingsConfigDict(env_prefix="RAW_STREAM_")

    service_name: str = "raw_stream"
    host: str = "0.0.0.0"
    port: int = 8770

    # "cam1=/app/infer/vid61.MOV,cam2=/app/infer/vid62.MOV"
    sources: str = ""

    fps: float = 10.0
    jpeg_quality: int = 60
    max_dim: int = 540  # downscale longest side before encode; <=0 keeps original

    def source_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.sources.split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            dev, src = item.split("=", 1)
            dev, src = dev.strip(), src.strip()
            if dev and src:
                out[dev] = src
        return out


settings = Settings()
