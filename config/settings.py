from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/lutheran_db"
    primary_search_version: str = "WEB"
    chroma_db_path: str = "./.chroma"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
