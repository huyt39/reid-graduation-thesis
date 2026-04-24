from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "gateway"
    service_port: int = 8080

    # JWT
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Upstream services
    streaming_url: str = "ws://streaming:8765"
    query_service_url: str = "http://query_service:8090"

    # Rate limiting (requests per minute per client IP)
    rate_limit_rpm: int = 120

    # Bootstrap admin credentials (for initial login)
    admin_username: str = "admin"
    admin_password: str = "admin"

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="GATEWAY_", extra="ignore",
    )


settings = Settings()
