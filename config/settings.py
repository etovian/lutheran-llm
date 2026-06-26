from typing import Literal
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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
