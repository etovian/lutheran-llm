from unittest.mock import MagicMock, patch
from database.connection import get_engine, check_connection

def test_get_engine():
    """Verify that get_engine instantiates create_engine with settings database_url."""
    with patch("database.connection.create_engine") as mock_create_engine:
        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql://test_url"
        
        get_engine(settings=mock_settings)
        mock_create_engine.assert_called_once_with("postgresql://test_url")

def test_check_connection_success():
    """Verify that check_connection returns True when connection succeeds."""
    mock_engine = MagicMock()
    mock_connection = mock_engine.connect.return_value.__enter__.return_value
    mock_connection.execute.return_value = True
    
    assert check_connection(mock_engine) is True

def test_check_connection_failure():
    """Verify that check_connection returns False when connection raises an exception."""
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = Exception("Connection failed")
    
    assert check_connection(mock_engine) is False
