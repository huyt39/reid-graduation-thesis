from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "vllm_service"
    service_port: int = 8100

    # Upstream OpenAI-compatible LLM endpoint
    llm_base_url: str = "http://vllm:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_api_key: str = ""
    llm_timeout_seconds: float = 30.0

    # Generation defaults
    temperature: float = 0.0
    max_tokens: int = 512

    # Readiness behaviour
    require_llm_for_ready: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="VLLM_", extra="ignore",
    )


settings = Settings()
