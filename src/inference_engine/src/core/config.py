from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    service_name: str = "inference_engine"
    service_port: int = 8000
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
