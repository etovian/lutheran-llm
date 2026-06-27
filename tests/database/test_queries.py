from unittest.mock import MagicMock
import pytest
from sqlalchemy.exc import OperationalError
from database.queries import fetch_parallel_translations

def test_fetch_parallel_translations():
    """Verify that parallel translations are queried and returned correctly."""
    mock_engine = MagicMock()
    mock_connection = mock_engine.connect.return_value.__enter__.return_value
    
    mock_result_tx = MagicMock()
    mock_result_tx.mappings.return_value.all.return_value = [
        {"version_code": "WEB", "text": "We maintain therefore that a man is justified..."},
        {"version_code": "KJV", "text": "Therefore we conclude that a man is justified..."}
    ]
    mock_connection.execute.return_value = mock_result_tx
    
    res = fetch_parallel_translations(mock_engine, verse_id=1)
    
    assert res["WEB"] == "We maintain therefore that a man is justified..."
    assert res["KJV"] == "Therefore we conclude that a man is justified..."


def test_fetch_parallel_translations_empty():
    """Verify that a non-existent verse_id returns an empty dictionary."""
    mock_engine = MagicMock()
    mock_connection = mock_engine.connect.return_value.__enter__.return_value
    
    mock_result_tx = MagicMock()
    mock_result_tx.mappings.return_value.all.return_value = []
    mock_connection.execute.return_value = mock_result_tx
    
    res = fetch_parallel_translations(mock_engine, verse_id=9999)
    assert res == {}


def test_fetch_parallel_translations_exception():
    """Verify that database errors are logged and re-raised."""
    mock_engine = MagicMock()
    mock_connection = mock_engine.connect.return_value.__enter__.return_value
    mock_connection.execute.side_effect = OperationalError("SELECT", {}, Exception("DB failure"))
    
    with pytest.raises(OperationalError):
        fetch_parallel_translations(mock_engine, verse_id=1)


def test_fetch_single_translation(db_engine):
    """fetch_single_translation returns verse text for a specific version."""
    from database.queries import fetch_single_translation
    from sqlalchemy import text
    # Seed a known verse
    with db_engine.connect() as conn:
        conn.execute(text("INSERT INTO book (book_id, name, testament) VALUES (99, 'TestBook', 'NT') ON CONFLICT DO NOTHING"))
        conn.execute(text("INSERT INTO verse (verse_id, book_id, chapter, verse_number, original_verse, address_code) VALUES (9901, 99, 1, 1, 'original', 'TST 1:1') ON CONFLICT DO NOTHING"))
        conn.execute(text("INSERT INTO verse_translation (verse_id, version_code, text) VALUES (9901, 'WEB', 'Test web text') ON CONFLICT DO NOTHING"))
        conn.execute(text("INSERT INTO verse_translation (verse_id, version_code, text) VALUES (9901, 'KJV', 'Test kjv text') ON CONFLICT DO NOTHING"))
        conn.commit()

    result = fetch_single_translation(db_engine, 9901, "KJV")
    assert result == "Test kjv text"

    result_missing = fetch_single_translation(db_engine, 9901, "MKJV")
    assert result_missing == ""
