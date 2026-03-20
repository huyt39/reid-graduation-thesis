from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    service_name: str = "worker"
    poll_interval_s: float = 1.0
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
