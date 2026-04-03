from pydantic_settings import BaseSettings


class EdgeSettings(BaseSettings):
    source_url: str = "data/video.mp4"
    device_id: str = "edge_device_1"
    model_path: str = "yolo11n.pt"
    yolo_conf_threshold: float = 0.25
    kafka_bootstrap_servers: str = "localhost:29092"
    reid_topic: str = "reid_input"
    schema_path: str = "../contracts/reid_input_v2.avsc"
    pre_skip_rate: int = 2
    v_good_threshold: float = 0.7
    v_mid_threshold: float = 0.4
    post_skip_good: int = 2
    post_skip_mid: int = 3
    post_skip_bad: int = 5
    drop_floor: float = 0.15

    class Config:
        env_prefix = "EDGE_"
