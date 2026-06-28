from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Settings class to manage application configuration.
    Loads configurations from environment variables or a local .env file.
    """
    database_url: str = "postgresql+psycopg://user:password@localhost:5432/lutheran_db"
    primary_search_version: Literal["WEB", "KJV", "MKJV"] = "WEB"
    chroma_db_path: str = "./.chroma"
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_predict: int = 512
    ollama_temperature: float = 0.0
    ollama_num_ctx: int = 2048
    rag_confessional_k: int = 5
    rag_biblical_k: int = 10
    rag_biblical_max_pool: int = 50
    rag_biblical_distance_threshold: float = 1.0
    llm_provider: Literal["ollama", "groq"] = "ollama"
    groq_api_key: Optional[str] = None
    groq_model: str = "llama3-8b-8192"
    groq_temperature: float = 0.0
    groq_max_tokens: int = 512

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
