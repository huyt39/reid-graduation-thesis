from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "edge"
    source_url: str = "0"
    device_id: str = "edge_device_1"
    model_path: str = "best.pt"
    demo_mode: bool = False
    yolo_conf_threshold: float = 0.25
    yolo_imgsz: int = 1280
    detect_every_n_frames: int = 1
    kafka_bootstrap_servers: str = "localhost:29092"
    reid_topic: str = "reid_input"
    schema_path: str = "src/contracts/reid_input.avsc"
    jpeg_quality: int = 90
    max_encode_dim: int = 0
    preview_enabled: bool = False
    preview_topic: str = "edge_preview"
    preview_fps: float = 12.0
    preview_jpeg_quality: int = 60
    preview_max_encode_dim: int = 720
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
    always_send_conf_threshold: float = 0.55
    always_send_visibility_threshold: float = 0.72
    always_send_min_area_ratio: float = 0.0075
    always_send_max_overlap_ratio: float = 0.35
    always_send_min_cutoff_score: float = 0.6
    poll_interval_s: float = 0.01
    log_every_n_processed_frames: int = 100
    log_every_n_frames: int = 30
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="EDGE_",
        extra="ignore",
    )


settings = Settings()
