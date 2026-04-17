from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "query_service"
    service_port: int = 8090

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "reid_production"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_dim: int = 512

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Upstream services
    vllm_service_url: str = "http://vllm_service:8100"
    inference_engine_url: str = "http://inference_engine:8000"

    # MinIO (for presigned URLs)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="QUERY_", extra="ignore",
    )


settings = Settings()
