from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "streaming"
    service_port: int = 8765

    # Kafka
    kafka_bootstrap_servers: str = "localhost:29092"
    output_topic: str = "reid_output"
    input_topic: str = "reid_input"
    consumer_group: str = "streaming_consumer_group"
    raw_consumer_group: str = "streaming_raw_consumer_group"
    schema_path: str = "src/contracts/reid_output.avsc"
    input_schema_path: str = "src/contracts/reid_input.avsc"
    max_poll_records: int = 50

    # WebSocket
    websocket_max_connections: int = 100
    broadcast_semaphore: int = 20

    # Image encoding
    jpeg_quality: int = 75

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="STREAMING_", extra="ignore",
    )


settings = Settings()
