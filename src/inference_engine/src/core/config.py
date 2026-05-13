from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "inference_engine"
    service_port: int = 8000

    # Model weight paths
    osnet_weights: str = "src/assets/models/osnet/model.pth.tar-150"
    lmbn_weights: str = ""  # optional, leave empty to skip
    efficientnet_weights: str = "src/assets/models/efficientnet/best_model.pth"
    # Multi-attribute classifier (8 PA-100K tasks on shared EfficientNet-B0 backbone).
    # When loaded, takes priority over `efficientnet_weights` for /gender/classify.
    multi_attr_weights: str = "src/assets/models/multi_attr/best_model_multi_attr_b0.pth"
    # Standalone gender classifier trained on PETA (88% val_acc). When loaded,
    # overrides the gender head of multi_attr_weights for all gender predictions.
    standalone_gender_weights: str = "src/assets/models/gender/gender_model.pth"

    # Inference
    device: str = "auto"  # "auto", "cuda", "cpu"
    embedding_dim: int = 512
    max_batch_size: int = 32
    batch_timeout_ms: int = 10

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="INFERENCE_", extra="ignore",
    )


settings = Settings()
