from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "reid_worker"
    poll_interval_s: float = 0.2
    log_every_n_messages: int = 25

    kafka_bootstrap_servers: str = "localhost:29092"
    input_topic: str = "reid_input"
    output_topic: str = "reid_output"
    consumer_group: str = "reid_worker_group"
    schema_path: str = "src/contracts/reid_input.avsc"
    output_schema_path: str = "src/contracts/reid_output.avsc"

    model_service_url: str = "http://localhost:8000"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_dim: int = 512
    similarity_threshold: float = 0.70
    match_margin: float = 0.02
    spatial_reuse_threshold: float = 0.70
    momentum: float = 0.8
    max_gallery_size: int = 8
    recent_person_reuse_enabled: bool = True
    recent_person_reuse_seconds: float = 2.5
    recent_person_reuse_min_iou: float = 0.2
    recent_person_reuse_max_center_distance_ratio: float = 0.75

    tracklet_min_entries: int = 8
    tracklet_max_entries: int = 60
    tracklet_window_seconds: float = 3.0
    tracklet_stale_seconds: float = 5.0

    topk_k: int = 5
    topk_min_temporal_gap: int = 3
    overlap_lambda: float = 0.3
    min_high_quality_frames: int = 3
    high_quality_threshold: float = 0.6

    gamma: float = 0.5

    promote_v_threshold: float = 0.6
    promote_consistency_threshold: float = 0.7

    tentative_max_attempts: int = 5
    tentative_fallback_enabled: bool = True

    update_v_threshold: float = 0.6
    update_consistency_threshold: float = 0.7
    update_min_tracklet_len: int = 5
    update_sim_threshold: float = 0.5
    soft_match_threshold: float = 0.72

    track_high_thresh: float = 0.7
    track_low_thresh: float = 0.35
    match_thresh: float = 0.3
    new_track_thresh: float = 0.82
    track_buffer: int = 30
    fuse_score: bool = True

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "reid_production"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    person_id_seq_key: str = "reid:seq:person_id"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"

    # Gender voting
    gender_person_threshold: float = 0.7
    gender_flip_threshold: float = 0.85

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WORKER_",
        extra="ignore",
    )


settings = Settings()
