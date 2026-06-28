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
    
    # db_lookup_func returns a string (the verse text) directly
    mock_db_lookup_func = MagicMock()
    mock_db_lookup_func.return_value = "For we maintain that a man is justified by faith"
    
    ctx = retrieve_context(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        query="justified",
        embed_model=mock_embed_model,
        db_lookup_func=mock_db_lookup_func
    )
    
    assert "confessional" in ctx
    assert len(ctx["confessional"]) == 1
    assert ctx["confessional"][0]["text"] == "Freely justified for Christ's sake"
    assert ctx["confessional"][0]["citation"] == "AC IV, 1"
    
    assert "scriptures" in ctx
    assert len(ctx["scriptures"]) == 1
    assert ctx["scriptures"][0]["verse_id"] == 1
    assert ctx["scriptures"][0]["primary_translation"] == "WEB"
    assert ctx["scriptures"][0]["cached_text"] == "For we maintain that a man is justified by faith"
    assert "translations" not in ctx["scriptures"][0]  # old key gone
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
        db_lookup_func=lambda eng, vid: "some text"
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
        return "some text"
        
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
        1: "First verse WEB text",
        2: "Second verse WEB text"
    }
    
    def mock_db_lookup(eng, vid):
        return mock_db_res_map.get(vid, "")
        
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
    assert ctx["scriptures"][0]["verse_id"] == 1
    assert ctx["scriptures"][0]["primary_translation"] == "WEB"
    assert ctx["scriptures"][0]["cached_text"] == "First verse WEB text"
    assert "translations" not in ctx["scriptures"][0]
    assert "lexicon" not in ctx["scriptures"][0]
    
    assert ctx["scriptures"][1]["citation"] == "Ephesians 2:8"
    assert ctx["scriptures"][1]["verse_id"] == 2
    assert ctx["scriptures"][1]["primary_translation"] == "WEB"
    assert ctx["scriptures"][1]["cached_text"] == "Second verse WEB text"
    assert "translations" not in ctx["scriptures"][1]
    assert "lexicon" not in ctx["scriptures"][1]


def test_retrieve_context_respects_primary_translation():
    """retrieve_context passes primary_translation to fetch_single_translation."""
    from unittest.mock import patch, MagicMock
    from pipeline.orchestrator import retrieve_context

    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed = MagicMock()
    mock_embed.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2])

    mock_chroma.get_collection.return_value.query.return_value = {
        "documents": [[]],
        "metadatas": [[{"verse_id": 42, "book_name": "John", "chapter": 3, "verse_number": 16, "address_code": "JHN 3:16"}]]
    }

    with patch("pipeline.orchestrator.fetch_single_translation", return_value="Car Dieu a tant aimé") as mock_fetch:
        result = retrieve_context(mock_chroma, mock_db, "query", mock_embed, primary_translation="KJV")

    mock_fetch.assert_called_once_with(mock_db, 42, "KJV")
    assert result["scriptures"][0]["primary_translation"] == "KJV"
    assert result["scriptures"][0]["cached_text"] == "Car Dieu a tant aimé"


def test_retrieve_context_filters_by_distance_threshold():
    """Verify that retrieve_context filters scriptures using rag_biblical_distance_threshold."""
    from unittest.mock import MagicMock
    
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value.tolist.return_value = [0.1, 0.2]
    
    mock_conf_collection = MagicMock()
    mock_conf_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    
    mock_bib_collection = MagicMock()
    mock_bib_collection.query.return_value = {
        "documents": [["Verse 1", "Verse 2"]],
        "metadatas": [[
            {"verse_id": 1, "book_name": "Romans", "chapter": 3, "verse_number": 28, "address_code": "ROM_3_28"},
            {"verse_id": 2, "book_name": "Ephesians", "chapter": 2, "verse_number": 8, "address_code": "EPH_2_8"}
        ]],
        "distances": [[0.5, 1.5]]  # 0.5 is <= 1.0 (keeps it), 1.5 is > 1.0 (filters it)
    }
    
    mock_chroma.get_collection.side_effect = lambda name: (
        mock_conf_collection if name == "confessional_collection" else mock_bib_collection
    )
    
    ctx = retrieve_context(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        query="justified",
        embed_model=mock_embed_model,
        db_lookup_func=lambda eng, vid: f"Text {vid}"
    )
    
    # Only verse 1 should be returned because distance 1.5 exceeds the default 1.0 threshold
    assert len(ctx["scriptures"]) == 1
    assert ctx["scriptures"][0]["verse_id"] == 1
    assert ctx["scriptures"][0]["distance"] == 0.5


