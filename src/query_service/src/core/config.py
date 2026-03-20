from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    service_name: str = "query_service"
    service_port: int = 8090
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
