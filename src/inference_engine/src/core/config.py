from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "inference_engine"
    service_port: int = 8000

    # Model weight paths
    osnet_weights: str = "src/assets/models/osnet/model.pth.tar-150"
    # OSNet-AIN (domain-generalization variant): the embedding that generalized best
    # to cross-view on the held-out 51/52 set (see MULTI_CAMERA.md). Empty to disable.
    osnet_ain_weights: str = "src/assets/models/osnet_ain/osnet_ain_msmt17.pth"
    osnet_onnx_path: str = ""
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

    # Triton backend (when enabled, skips loading OSNet/multi_attr PyTorch
    # models in-process; the standalone gender + legacy gender models still
    # load locally because they aren't exported to Triton yet).
    use_triton: bool = False
    triton_url: str = "triton:8000"

    # PAR preprocessing: aspect-preserving letterbox (224x224) for the
    # multi-attribute classifier crop.
    # Default OFF: the current PA-100K classifier weights were trained with
    # naive Resize((224, 224)) (see Yolo-for-Edge-Devices/huy_backup/
    # MultiAttr_EfficientNetB0.py val_transform). Switching to letterbox at
    # inference creates a train/test preprocessing mismatch and degrades
    # accuracy in practice. Re-enable only after retraining the classifier
    # with letterbox preprocessing.
    par_letterbox: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="INFERENCE_", extra="ignore",
    )


settings = Settings()
