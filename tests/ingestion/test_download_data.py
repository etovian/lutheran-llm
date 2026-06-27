from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from ingestion.download_data import download_file, download_all, BIBLE_FILES, BOC_FILES

def test_download_file_success(tmp_path):
    """Verify that download_file successfully requests a URL and saves its contents using Path and Session."""
    url = "https://example.com/test.json"
    dest = tmp_path / "test.json"
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"key": "value"}'
    
    # Test with standard requests.get (no session)
    with patch("requests.get", return_value=mock_response) as mock_get:
        download_file(url, dest)
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == url
        assert "User-Agent" in kwargs["headers"]
        assert dest.read_text(encoding="utf-8") == '{"key": "value"}'

    # Test with custom Session
    dest_session = tmp_path / "test_session.json"
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    
    download_file(url, dest_session, session=mock_session)
    mock_session.get.assert_called_once()
    args, kwargs = mock_session.get.call_args
    assert args[0] == url
    assert "User-Agent" in kwargs["headers"]
    assert dest_session.read_text(encoding="utf-8") == '{"key": "value"}'

def test_download_file_retries_on_failure(tmp_path):
    """Verify that download_file performs retries upon experiencing an HTTP error, utilizing tmp_path."""
    url = "https://example.com/test.json"
    dest = tmp_path / "test.json"
    
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 500
    mock_response_fail.raise_for_status.side_effect = Exception("HTTP Error")
    
    with patch("requests.get", return_value=mock_response_fail) as mock_get, \
         patch("time.sleep") as mock_sleep:
        
        with pytest.raises(Exception):
            download_file(url, dest, max_retries=3, backoff_factor=0.1)
        
        # Should be called 4 times (1 initial attempt + 3 retries)
        assert mock_get.call_count == 4

def test_download_all_lists(tmp_path):
    """Verify that download_all calls download_file for all specified Bibles and Book of Concord files using Path."""
    with patch("ingestion.download_data.download_file") as mock_download_file:
        download_all(dest_dir=tmp_path)
        
        # Verify download_file was called with expected arguments.
        # Since we use requests.Session() inside download_all, we check the url and path.
        calls = mock_download_file.call_args_list
        assert len(calls) == len(BIBLE_FILES) + len(BOC_FILES)
        
        # Check specific Bible and BOC file call patterns
        called_args = [(call[0][0], call[0][1]) for call in calls]
        
        expected_bible_paths = [
            (
                "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json/KJV.json",
                tmp_path / "KJV.json"
            ),
            (
                "https://raw.githubusercontent.com/scrollmapper/bible_databases/master/formats/json/MKJV.json",
                tmp_path / "MKJV.json"
            ),
            (
                "https://raw.githubusercontent.com/seven1m/open-bibles/master/eng-web.usfx.xml",
                tmp_path / "eng-web.usfx.xml"
            )
        ]
        for url, path in expected_bible_paths:
            assert (url, path) in called_args
            
        expected_boc_paths = [
            (
                "https://raw.githubusercontent.com/remysheppard/lutheran-confessions/main/content/boc/apology/_index.md",
                tmp_path / "boc" / "apology" / "_index.md"
            ),
            (
                "https://raw.githubusercontent.com/remysheppard/lutheran-confessions/main/content/boc/formula/epitome/election.md",
                tmp_path / "boc" / "formula" / "epitome" / "election.md"
            )
        ]
        for url, path in expected_boc_paths:
            assert (url, path) in called_args
            
        # Assert that session parameter is passed as a kwarg and is not None
        for call in calls:
            assert "session" in call[1]
            assert call[1]["session"] is not None
