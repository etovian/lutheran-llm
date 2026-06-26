from config.settings import Settings

def test_settings_load_from_env(monkeypatch):
    """
    Test that Settings correctly overrides default values when corresponding
    environment variables are set.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://test_user:test_pass@localhost:5432/test_db")
    monkeypatch.setenv("PRIMARY_SEARCH_VERSION", "WEB")
    monkeypatch.setenv("CHROMA_DB_PATH", "./test_chroma")
    
    settings = Settings()
    
    assert settings.database_url == "postgresql://test_user:test_pass@localhost:5432/test_db"
    assert settings.primary_search_version == "WEB"
    assert settings.chroma_db_path == "./test_chroma"


def test_settings_default_values(monkeypatch):
    """
    Test that Settings loads default values correctly when no environment
    variables are present.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PRIMARY_SEARCH_VERSION", raising=False)
    monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
    
    settings = Settings()
    assert settings.database_url == "postgresql://postgres:postgres@localhost:5432/lutheran_db"
    assert settings.primary_search_version == "WEB"
    assert settings.chroma_db_path == "./.chroma"
