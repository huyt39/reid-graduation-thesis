from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_prefix="RAW_STREAM_")

    service_name: str = "raw_stream"
    host: str = "0.0.0.0"
    port: int = 8770

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
