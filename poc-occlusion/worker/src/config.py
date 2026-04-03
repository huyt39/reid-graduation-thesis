from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    kafka_bootstrap_servers: str = "localhost:29092"
    input_topic: str = "reid_input"
    consumer_group: str = "reid_worker_group"
    schema_path: str = "../contracts/reid_input_v2.avsc"
    model_service_url: str = "http://localhost:8000"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_host: str = "localhost"
    redis_port: int = 6379
    embedding_dim: int = 512
    similarity_threshold: float = 0.70
    momentum: float = 0.8

    # Tracklet buffer
    tracklet_min_entries: int = 8
    tracklet_max_entries: int = 60
    tracklet_window_seconds: float = 3.0
    tracklet_stale_seconds: float = 5.0

    # Top-K selection
    topk_k: int = 5
    topk_min_temporal_gap: int = 3
    overlap_lambda: float = 0.3
    min_high_quality_frames: int = 3
    high_quality_threshold: float = 0.6

    # Embedding aggregation
    gamma: float = 0.5  # overlap penalty in weight: w = v * (1 - gamma * overlap)

    # Promote tentative policy
    promote_v_threshold: float = 0.6
    promote_consistency_threshold: float = 0.7

    # Gated canonical update
    update_v_threshold: float = 0.6
    update_consistency_threshold: float = 0.7
    update_min_tracklet_len: int = 5
    update_sim_threshold: float = 0.5

    # BYTETracker
    track_high_thresh: float = 0.7
    track_low_thresh: float = 0.35
    match_thresh: float = 0.3
    new_track_thresh: float = 0.82
    track_buffer: int = 30
    fuse_score: bool = True

    class Config:
        env_prefix = "WORKER_"
