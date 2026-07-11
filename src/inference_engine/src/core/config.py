from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "inference_engine"
    service_port: int = 8000

    # Model weight paths
    osnet_weights: str = "src/assets/models/osnet/model.pth.tar-150"
    # OSNet-AIN
    osnet_ain_weights: str = "src/assets/models/osnet_ain/osnet_ain_msmt17.pth"
    osnet_onnx_path: str = ""
    lmbn_weights: str = ""
    efficientnet_weights: str = "src/assets/models/efficientnet/best_model.pth"
    multi_attr_weights: str = "src/assets/models/multi_attr/best_model_multi_attr_b0.pth"
    standalone_gender_weights: str = "src/assets/models/gender/gender_model.pth"
    # EfficientNet-B0 gender classifier (efficientnet_pytorch layout). When present it
    # takes priority over the standalone/multi-attr gender heads for /gender/classify.
    effb0_gender_weights: str = "src/assets/models/gender/gender_effb0.pth"

    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    embedding_dim: int = 512
    max_batch_size: int = 32
    batch_timeout_ms: int = 10

    # Pedestrian Attribute Recognition preprocessing: aspect-preserving letterbox (224x224) for the
    # multi-attribute classifier crop.
    par_letterbox: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="INFERENCE_", extra="ignore",
    )


settings = Settings()
