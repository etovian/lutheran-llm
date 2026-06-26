from config.settings import Settings

def test_settings_load_from_env(monkeypatch):
    """
    Test that Settings correctly overrides default values when corresponding
    environment variables are set.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://test_user:test_pass@localhost:5432/test_db")
    monkeypatch.setenv("PRIMARY_SEARCH_VERSION", "WEB")
    monkeypatch.setenv("CHROMA_DB_PATH", "./test_chroma")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11435")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "200")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.5")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "2048")
    
    settings = Settings()
    
    assert settings.database_url == "postgresql://test_user:test_pass@localhost:5432/test_db"
    assert settings.primary_search_version == "WEB"
    assert settings.chroma_db_path == "./test_chroma"
    assert settings.ollama_model == "mistral"
    assert settings.ollama_base_url == "http://localhost:11435"
    assert settings.ollama_num_predict == 200
    assert settings.ollama_temperature == 0.5
    assert settings.ollama_num_ctx == 2048


def test_settings_default_values(monkeypatch):
    """
    Test that Settings loads default values correctly when no environment
    variables are present.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PRIMARY_SEARCH_VERSION", raising=False)
    monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    
    settings = Settings()
    assert settings.database_url == "postgresql+psycopg://user:password@localhost:5432/lutheran_db"
    assert settings.primary_search_version == "WEB"
    assert settings.chroma_db_path == "./.chroma"
    assert settings.ollama_model == "llama3"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_num_predict == 150
    assert settings.ollama_temperature == 0.0
    assert settings.ollama_num_ctx == 1024
