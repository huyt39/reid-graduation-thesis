from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "gateway"
    service_port: int = 8080

    # Upstream services
    streaming_url: str = "ws://streaming:8765"
    query_service_url: str = "http://query_service:8090"

    # Rate limiting (requests per minute per client IP)
    rate_limit_rpm: int = 120

    # CORS — comma-separated list of allowed browser origins
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="GATEWAY_", extra="ignore",
    )


settings = Settings()
