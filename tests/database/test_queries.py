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

