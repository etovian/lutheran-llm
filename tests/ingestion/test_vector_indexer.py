from unittest.mock import MagicMock
from ingestion.vector_indexer import VectorIndexer

def test_indexing_confessional_chunk():
    """Verify that a confessional chunk is correctly embedded and added to the confessional collection."""
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    
    indexer = VectorIndexer(chroma_client=mock_chroma_client)
    
    chunk = {
        "text": "Freely justified for Christ's sake",
        "book": "Augsburg Confession",
        "article_id": "AC_IV",
        "paragraph_number": 2,
        "citation": "AC IV, Paragraph 2"
    }
    
    indexer.index_confessional(chunk)
    
    mock_chroma_client.get_collection.assert_called_with("confessional_collection")
    mock_collection.add.assert_called_once()
    call_args = mock_collection.add.call_args[1]
    assert call_args["documents"] == ["Freely justified for Christ's sake"]
    assert call_args["metadatas"] == [{
        "book": "Augsburg Confession",
        "article_id": "AC_IV",
        "paragraph_number": 2,
        "citation": "AC IV, Paragraph 2"
    }]
    assert call_args["ids"] == ["Augsburg Confession_AC_IV_2"]

def test_indexing_biblical_verse():
    """Verify that a biblical verse is correctly embedded and added to the biblical collection."""
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    
    indexer = VectorIndexer(chroma_client=mock_chroma_client)
    
    verse = {
        "text": "For by grace you have been saved through faith...",
        "verse_id": 28145,
        "address_code": "EPH_2_8",
        "book_name": "Ephesians",
        "chapter": 2,
        "verse_number": 8
    }
    
    indexer.index_biblical(verse)
    
    mock_chroma_client.get_collection.assert_called_with("biblical_collection")
    mock_collection.add.assert_called_once()
    call_args = mock_collection.add.call_args[1]
    assert call_args["documents"] == ["For by grace you have been saved through faith..."]
    assert call_args["metadatas"] == [{
        "verse_id": 28145,
        "address_code": "EPH_2_8",
        "book_name": "Ephesians",
        "chapter": 2,
        "verse_number": 8
    }]
    assert call_args["ids"] == ["28145"]
