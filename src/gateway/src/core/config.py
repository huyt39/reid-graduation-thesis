from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    service_name: str = "gateway"
    service_port: int = 8080
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
