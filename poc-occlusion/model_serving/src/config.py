from pydantic_settings import BaseSettings


class ModelServingSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    model_name: str = "osnet"
    weights_dir: str = "weights"
    gem_p: float = 3.0
    image_height: int = 256
    image_width: int = 128

    class Config:
        env_prefix = "MODEL_"
