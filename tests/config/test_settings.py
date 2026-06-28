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
    
    settings = Settings(_env_file=None)
    
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
    monkeypatch.delenv("RAG_CONFESSIONAL_K", raising=False)
    monkeypatch.delenv("RAG_BIBLICAL_K", raising=False)
    
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    
    settings = Settings(_env_file=None)
    assert settings.database_url == "postgresql+psycopg://user:password@localhost:5432/lutheran_db"
    assert settings.primary_search_version == "WEB"
    assert settings.chroma_db_path == "./.chroma"
    assert settings.ollama_model == "llama3"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_num_predict == 512
    assert settings.ollama_temperature == 0.0
    assert settings.ollama_num_ctx == 2048
    assert settings.rag_confessional_k == 2
    assert settings.rag_biblical_k == 10
    assert settings.llm_provider == "ollama"
    assert settings.groq_api_key is None
    assert settings.groq_model == "llama3-8b-8192"
    assert settings.groq_temperature == 0.0
    assert settings.groq_max_tokens == 512


def test_settings_rag_k_values(monkeypatch):
    """
    Test that RAG settings correctly load configuration values from environment variables.
    """
    monkeypatch.setenv("RAG_CONFESSIONAL_K", "5")
    monkeypatch.setenv("RAG_BIBLICAL_K", "2")
    settings = Settings(_env_file=None)
    assert settings.rag_confessional_k == 5
    assert settings.rag_biblical_k == 2


def test_settings_groq_overrides(monkeypatch):
    """
    Test that Groq settings are loaded from environment variables.
    """
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test123")
    monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
    monkeypatch.setenv("GROQ_TEMPERATURE", "0.7")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "1024")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "groq"
    assert settings.groq_api_key == "gsk_test123"
    assert settings.groq_model == "mixtral-8x7b-32768"
    assert settings.groq_temperature == 0.7
    assert settings.groq_max_tokens == 1024


