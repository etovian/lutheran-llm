from unittest.mock import MagicMock
import pytest
from pipeline.orchestrator import retrieve_context

def test_retrieve_context():
    """Verify that query context is retrieved from Chroma collections and relational DB correctly."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed_model = MagicMock()
    
    # Mock embedding generation
    mock_embed_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    
    # Mock Chroma collections query returns
    # First call: confessional collection query
    # Second call: biblical collection query
    mock_conf_collection = MagicMock()
    mock_conf_collection.query.return_value = {
        "documents": [["Freely justified for Christ's sake"]],
        "metadatas": [[{"citation": "AC IV, 1"}]]
    }
    
    mock_bib_collection = MagicMock()
    mock_bib_collection.query.return_value = {
        "documents": [["We maintain that a man is justified..."]],
        "metadatas": [[{"verse_id": 1, "address_code": "ROM_3_28"}]]
    }
    
    mock_chroma.get_collection.side_effect = lambda name: (
        mock_conf_collection if name == "confessional_collection" else mock_bib_collection
    )
    
    # Mock DB queries.py response
    mock_db_res = {
        "translations": {"WEB": "We maintain that a man is justified...", "KJV": "Therefore we conclude..."},
        "lexicon": [{"word_text": "λογιζόμεθα", "strongs_number": "G3049", "definition": "To reckon"}]
    }
    
    ctx = retrieve_context(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        query="justified",
        embed_model=mock_embed_model,
        db_lookup_func=lambda eng, vid: mock_db_res
    )
    
    assert "confessional" in ctx
    assert len(ctx["confessional"]) == 1
    assert ctx["confessional"][0]["text"] == "Freely justified for Christ's sake"
    assert ctx["confessional"][0]["citation"] == "AC IV, 1"
    
    assert "scripture" in ctx
    assert "translations" in ctx["scripture"]
    assert ctx["scripture"]["translations"]["WEB"] == "We maintain that a man is justified..."
    assert ctx["scripture"]["translations"]["KJV"] == "Therefore we conclude..."
    assert ctx["scripture"]["lexicon"][0]["word_text"] == "λογιζόμεθα"


def test_retrieve_context_empty():
    """Verify that when ChromaDB returns no results, retrieve_context handles it and returns empty values."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [[]],
        "metadatas": [[]]
    }
    mock_chroma.get_collection.return_value = mock_collection
    
    ctx = retrieve_context(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        query="justified",
        embed_model=mock_embed_model,
        db_lookup_func=lambda eng, vid: {}
    )
    assert ctx == {"confessional": [], "scripture": {}}


def test_retrieve_context_missing_verse_id():
    """Verify that a missing 'verse_id' in scripture metadata bypasses database lookup."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    
    mock_conf_collection = MagicMock()
    mock_conf_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    
    mock_bib_collection = MagicMock()
    mock_bib_collection.query.return_value = {"documents": [[]], "metadatas": [[{"address_code": "ROM_3_28"}]]}
    
    mock_chroma.get_collection.side_effect = lambda name: (
        mock_conf_collection if name == "confessional_collection" else mock_bib_collection
    )
    
    db_called = False
    def mock_db_lookup(eng, vid):
        nonlocal db_called
        db_called = True
        return {}
        
    ctx = retrieve_context(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        query="justified",
        embed_model=mock_embed_model,
        db_lookup_func=mock_db_lookup
    )
    assert db_called is False
    assert ctx["scripture"] == {}


def test_retrieve_context_exception():
    """Verify that exceptions raised in retrieval are logged and re-raised."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed_model = MagicMock()
    mock_embed_model.encode.side_effect = Exception("Embedding model crash")
    
    with pytest.raises(Exception, match="Embedding model crash"):
        retrieve_context(mock_chroma, mock_db, "justified", mock_embed_model)
