import sys
from unittest.mock import patch, MagicMock
import pytest

@patch("streamlit.set_page_config")
@patch("streamlit.markdown")
@patch("streamlit.sidebar")
@patch("ui.app.load_db_engine")
@patch("ui.app.load_chroma_client")
@patch("ui.app.load_embedding_model")
def test_streamlit_app_initialization(
    mock_load_embed,
    mock_load_chroma,
    mock_load_db,
    mock_sidebar,
    mock_markdown,
    mock_set_page_config
):
    """Verify that importing the Streamlit app configures the page and attempts resource loads."""
    # Setup mocks
    mock_load_db.return_value = MagicMock()
    mock_load_chroma.return_value = MagicMock()
    mock_load_embed.return_value = MagicMock()
    
    # Reload/import the module to trigger execution
    if "ui.app" in sys.modules:
        del sys.modules["ui.app"]
    import ui.app
    
    # Assert st.set_page_config was called with expected title
    mock_set_page_config.assert_called_once_with(
        page_title="Lutheran Confessional Assistant",
        page_icon="⛪",
        layout="centered"
    )
    
    # Verify main title markdown was rendered
    title_rendered = any(
        "Lutheran Confessional Assistant" in call[0][0] 
        for call in mock_markdown.call_args_list
    )
    assert title_rendered is True
