from unittest.mock import MagicMock, patch
from ingestion.vector_indexer import VectorIndexer

@patch("ingestion.vector_indexer.SentenceTransformer")
def test_indexing_confessional_chunk(mock_transformer_cls):
    """Verify that a confessional chunk is correctly embedded and added to the confessional collection."""
    mock_model = MagicMock()
    mock_transformer_cls.return_value = mock_model
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
    
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
    assert call_args["embeddings"] == [[0.1, 0.2, 0.3]]
    assert call_args["metadatas"] == [{
        "book": "Augsburg Confession",
        "article_id": "AC_IV",
        "paragraph_number": 2,
        "citation": "AC IV, Paragraph 2"
    }]
    assert call_args["ids"] == ["Augsburg Confession_AC_IV_2"]


@patch("ingestion.vector_indexer.SentenceTransformer")
def test_indexing_biblical_verse(mock_transformer_cls):
    """Verify that a biblical verse is correctly embedded and added to the biblical collection."""
    mock_model = MagicMock()
    mock_transformer_cls.return_value = mock_model
    mock_model.encode.return_value.tolist.return_value = [[0.4, 0.5, 0.6]]
    
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
    assert call_args["embeddings"] == [[0.4, 0.5, 0.6]]
    assert call_args["metadatas"] == [{
        "verse_id": 28145,
        "address_code": "EPH_2_8",
        "book_name": "Ephesians",
        "chapter": 2,
        "verse_number": 8
    }]
    assert call_args["ids"] == ["28145"]


@patch("ingestion.vector_indexer.SentenceTransformer")
def test_indexing_confessional_batch(mock_transformer_cls):
    """Verify that a batch of confessional chunks is correctly embedded and added."""
    mock_model = MagicMock()
    mock_transformer_cls.return_value = mock_model
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    
    indexer = VectorIndexer(chroma_client=mock_chroma_client)
    
    chunks = [
        {
            "text": "First chunk text",
            "book": "Augsburg Confession",
            "article_id": "AC_IV",
            "paragraph_number": 1,
            "citation": "AC IV, Paragraph 1"
        },
        {
            "text": "Second chunk text",
            "book": "Augsburg Confession",
            "article_id": "AC_IV",
            "paragraph_number": 2,
            "citation": "AC IV, Paragraph 2"
        }
    ]
    
    indexer.index_confessional_batch(chunks)
    
    mock_chroma_client.get_collection.assert_called_with("confessional_collection")
    mock_collection.add.assert_called_once_with(
        documents=["First chunk text", "Second chunk text"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[
            {
                "book": "Augsburg Confession",
                "article_id": "AC_IV",
                "paragraph_number": 1,
                "citation": "AC IV, Paragraph 1"
            },
            {
                "book": "Augsburg Confession",
                "article_id": "AC_IV",
                "paragraph_number": 2,
                "citation": "AC IV, Paragraph 2"
            }
        ],
        ids=["Augsburg Confession_AC_IV_1", "Augsburg Confession_AC_IV_2"]
    )


@patch("ingestion.vector_indexer.SentenceTransformer")
def test_indexing_biblical_batch(mock_transformer_cls):
    """Verify that a batch of biblical verses is correctly embedded and added."""
    mock_model = MagicMock()
    mock_transformer_cls.return_value = mock_model
    mock_model.encode.return_value.tolist.return_value = [[0.5, 0.6], [0.7, 0.8]]
    
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    
    indexer = VectorIndexer(chroma_client=mock_chroma_client)
    
    verses = [
        {
            "text": "Verse 1",
            "verse_id": 1001,
            "address_code": "GEN_1_1",
            "book_name": "Genesis",
            "chapter": 1,
            "verse_number": 1
        },
        {
            "text": "Verse 2",
            "verse_id": 1002,
            "address_code": "GEN_1_2",
            "book_name": "Genesis",
            "chapter": 1,
            "verse_number": 2
        }
    ]
    
    indexer.index_biblical_batch(verses)
    
    mock_chroma_client.get_collection.assert_called_with("biblical_collection")
    mock_collection.add.assert_called_once_with(
        documents=["Verse 1", "Verse 2"],
        embeddings=[[0.5, 0.6], [0.7, 0.8]],
        metadatas=[
            {
                "verse_id": 1001,
                "address_code": "GEN_1_1",
                "book_name": "Genesis",
                "chapter": 1,
                "verse_number": 1
            },
            {
                "verse_id": 1002,
                "address_code": "GEN_1_2",
                "book_name": "Genesis",
                "chapter": 1,
                "verse_number": 2
            }
        ],
        ids=["1001", "1002"]
    )
