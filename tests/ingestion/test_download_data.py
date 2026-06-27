import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
from ingestion.download_data import download_file, download_all, BIBLE_FILES, BOC_FILES

def test_download_file_success():
    """Verify that download_file successfully requests a URL and saves its contents to the destination."""
    url = "https://example.com/test.json"
    dest = os.path.join("data", "test.json")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"key": "value"}'
    
    with patch("requests.get", return_value=mock_response) as mock_get, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("os.makedirs") as mock_makedirs:
        
        download_file(url, dest)
        
        # Verify directory creation for destination path
        mock_makedirs.assert_called_once_with("data", exist_ok=True)
        # Verify requests.get called with user agent headers
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == url
        assert "User-Agent" in kwargs["headers"]
        # Verify file written
        mock_file.assert_called_once_with(dest, "w", encoding="utf-8")
        mock_file().write.assert_called_once_with('{"key": "value"}')

def test_download_file_retries_on_failure():
    """Verify that download_file performs retries upon experiencing an HTTP error."""
    url = "https://example.com/test.json"
    dest = os.path.join("data", "test.json")
    
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 500
    mock_response_fail.raise_for_status.side_effect = Exception("HTTP Error")
    
    with patch("requests.get", return_value=mock_response_fail) as mock_get, \
         patch("time.sleep") as mock_sleep:
        
        with pytest.raises(Exception):
            download_file(url, dest, max_retries=3, backoff_factor=0.1)
        
        # Should be called 4 times (1 initial attempt + 3 retries)
        assert mock_get.call_count == 4

def test_download_all_lists():
    """Verify that download_all calls download_file for all specified Bibles and Book of Concord files."""
    with patch("ingestion.download_data.download_file") as mock_download_file:
        download_all(dest_dir="data")
        
        # Verify primary Bible files are downloaded
        expected_bible_urls = [
            "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json/KJV.json",
            "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json/MKJV.json",
            "https://raw.githubusercontent.com/seven1m/open-bibles/master/eng-web.usfx.xml"
        ]
        for url in expected_bible_urls:
            filename = url.split("/")[-1]
            mock_download_file.assert_any_call(url, os.path.join("data", filename))
            
        # Verify some specific Book of Concord files are downloaded
        mock_download_file.assert_any_call(
            "https://raw.githubusercontent.com/remysheppard/lutheran-confessions/main/content/boc/apology/_index.md",
            os.path.join("data", "boc", "apology", "_index.md")
        )
        mock_download_file.assert_any_call(
            "https://raw.githubusercontent.com/remysheppard/lutheran-confessions/main/content/boc/formula/epitome/election.md",
            os.path.join("data", "boc", "formula", "epitome", "election.md")
        )
        mock_download_file.assert_any_call(
            "https://raw.githubusercontent.com/remysheppard/lutheran-confessions/main/content/boc/small-catechism/ten-commandments.md",
            os.path.join("data", "boc", "small-catechism", "ten-commandments.md")
        )
