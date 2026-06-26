from config.settings import Settings

def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test_user:test_pass@localhost:5432/test_db")
    monkeypatch.setenv("PRIMARY_SEARCH_VERSION", "WEB")
    monkeypatch.setenv("CHROMA_DB_PATH", "./test_chroma")
    
    settings = Settings()
    
    assert settings.database_url == "postgresql://test_user:test_pass@localhost:5432/test_db"
    assert settings.primary_search_version == "WEB"
    assert settings.chroma_db_path == "./test_chroma"
