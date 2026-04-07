from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "edge"
    source_url: str = "0"
    device_id: str = "edge_device_1"
    model_path: str = "best.pt"
    yolo_conf_threshold: float = 0.25
    yolo_imgsz: int = 1280
    kafka_bootstrap_servers: str = "localhost:29092"
    reid_topic: str = "reid_input"
    schema_path: str = "src/contracts/reid_input.avsc"
    jpeg_quality: int = 70
    pre_skip_max_detected: int = 5
    pre_skip_max_empty: int = 30
    pre_skip_box_count_weight: float = 0.1
    pre_skip_criterion_scale: float = 0.1
    pre_skip_gray_size: tuple[int, int] = Field(default=(160, 90))
    v_good_threshold: float = 0.7
    v_mid_threshold: float = 0.4
    post_skip_good: int = 2
    post_skip_mid: int = 3
    post_skip_bad: int = 5
    drop_floor: float = 0.15
    poll_interval_s: float = 0.01
    log_every_n_processed_frames: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EDGE_",
        extra="ignore",
    )


settings = Settings()
