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
        "metadatas": [[{"verse_id": 1, "address_code": "ROM_3_28", "book_name": "Romans", "chapter": 3, "verse_number": 28}]]
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
    
    assert "scriptures" in ctx
    assert len(ctx["scriptures"]) == 1
    assert "translations" in ctx["scriptures"][0]
    assert ctx["scriptures"][0]["translations"]["WEB"] == "We maintain that a man is justified..."
    assert ctx["scriptures"][0]["translations"]["KJV"] == "Therefore we conclude..."
    assert ctx["scriptures"][0]["lexicon"][0]["word_text"] == "λογιζόμεθα"
    assert ctx["scriptures"][0]["citation"] == "Romans 3:28"


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
    assert ctx == {"confessional": [], "scriptures": []}


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
    assert ctx["scriptures"] == []


def test_retrieve_context_exception():
    """Verify that exceptions raised in retrieval are logged and re-raised."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed_model = MagicMock()
    mock_embed_model.encode.side_effect = Exception("Embedding model crash")
    
    with pytest.raises(Exception, match="Embedding model crash"):
        retrieve_context(mock_chroma, mock_db, "justified", mock_embed_model)


def test_retrieve_context_multiple_citations():
    """Verify that query context retrieves multiple scriptures from ChromaDB and relational database."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    
    mock_conf_collection = MagicMock()
    mock_conf_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    
    mock_bib_collection = MagicMock()
    mock_bib_collection.query.return_value = {
        "documents": [["First verse text", "Second verse text"]],
        "metadatas": [[
            {"verse_id": 1, "address_code": "ROM_3_28", "book_name": "Romans", "chapter": 3, "verse_number": 28},
            {"verse_id": 2, "address_code": "EPH_2_8", "book_name": "Ephesians", "chapter": 2, "verse_number": 8}
        ]]
    }
    
    mock_chroma.get_collection.side_effect = lambda name: (
        mock_conf_collection if name == "confessional_collection" else mock_bib_collection
    )
    
    mock_db_res_map = {
        1: {
            "translations": {"WEB": "First verse WEB", "KJV": "First verse KJV"},
            "lexicon": [{"word_text": "λογιζόμεθα", "strongs_number": "G3049", "definition": "To reckon"}]
        },
        2: {
            "translations": {"WEB": "Second verse WEB", "KJV": "Second verse KJV"},
            "lexicon": [{"word_text": "χάριτί", "strongs_number": "G5485", "definition": "grace"}]
        }
    }
    
    def mock_db_lookup(eng, vid):
        return mock_db_res_map.get(vid, {})
        
    ctx = retrieve_context(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        query="justified by grace",
        embed_model=mock_embed_model,
        db_lookup_func=mock_db_lookup
    )
    
    assert "scriptures" in ctx
    assert len(ctx["scriptures"]) == 2
    
    assert ctx["scriptures"][0]["citation"] == "Romans 3:28"
    assert ctx["scriptures"][0]["translations"]["WEB"] == "First verse WEB"
    assert ctx["scriptures"][0]["lexicon"][0]["word_text"] == "λογιζόμεθα"
    
    assert ctx["scriptures"][1]["citation"] == "Ephesians 2:8"
    assert ctx["scriptures"][1]["translations"]["WEB"] == "Second verse WEB"
    assert ctx["scriptures"][1]["lexicon"][0]["word_text"] == "χάριτί"
