from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/lutheran_db"
    primary_search_version: str = "WEB"
    chroma_db_path: str = "./.chroma"

    class Config:
        env_file = ".env"
