from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    llm_model: str = "qwen3-vl:8b"
    ollama_host: str = "http://localhost:11434"

    class Config:
        env_file = ".env"


settings = Settings()