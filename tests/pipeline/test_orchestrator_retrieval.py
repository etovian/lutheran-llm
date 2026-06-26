from unittest.mock import MagicMock
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
