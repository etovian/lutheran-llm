from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.orchestrator import run_orchestrator, format_deep_dive_details

@patch("pipeline.orchestrator.retrieve_context")
def test_run_orchestrator(mock_retrieve_context):
    """Verify that run_orchestrator retrieves context, formats prompt, and calls LLM correctly."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_llm = MagicMock()
    mock_embed_model = MagicMock()
    
    # Configure mock retrieve_context return values (no lexicon)
    mock_retrieve_context.return_value = {
        "confessional": [{"text": "Freely justified", "citation": "AC IV, 1"}],
        "scriptures": [{
            "citation": "Romans 3:28",
            "verse_id": 1,
            "primary_translation": "WEB",
            "cached_text": "Justified by faith",
        }]
    }
    
    # Configure mock LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Summary: We are justified by faith."
    mock_llm.invoke.return_value = mock_llm_response
    
    res = run_orchestrator(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        llm=mock_llm,
        query="How are we justified?",
        embed_model=mock_embed_model,
        primary_translation="WEB"
    )
    
    assert "Summary: We are justified by faith." in res
    
    mock_retrieve_context.assert_called_once_with(
        mock_chroma, mock_db, "How are we justified?", mock_embed_model,
        confessional_k=None, biblical_k=None, primary_translation="WEB"
    )
    
    # Verify LLM invoke was called with SystemMessage (containing context) and HumanMessage
    mock_llm.invoke.assert_called_once()
    messages = mock_llm.invoke.call_args[0][0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Freely justified" in messages[0].content
    assert "Justified by faith" in messages[0].content
    assert messages[1].content == "How are we justified?"


@patch("pipeline.orchestrator.retrieve_context")
def test_run_orchestrator_exception(mock_retrieve_context):
    """Verify that exceptions raised during orchestrator run are logged and re-raised."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_llm = MagicMock()
    mock_embed_model = MagicMock()
    
    mock_retrieve_context.return_value = {
        "confessional": [],
        "scriptures": []
    }
    
    mock_llm.invoke.side_effect = Exception("LLM connection timed out")
    
    with pytest.raises(Exception, match="LLM connection timed out"):
        run_orchestrator(mock_chroma, mock_db, mock_llm, "Query", mock_embed_model)


@patch("pipeline.orchestrator.detect_pastoral_crisis")
@patch("pipeline.orchestrator.retrieve_context")
def test_run_orchestrator_crisis_preemption(mock_retrieve_context, mock_detect_crisis):
    """Verify that run_orchestrator pre-empts normal RAG synthesis on crisis queries."""
    mock_detect_crisis.return_value = True
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_llm = MagicMock()
    mock_embed_model = MagicMock()
    
    res = run_orchestrator(mock_chroma, mock_db, mock_llm, "I feel so guilty", mock_embed_model)
    
    assert "comforting Gospel" in res or "pastor" in res
    mock_retrieve_context.assert_not_called()
    mock_llm.invoke.assert_not_called()


@patch("pipeline.orchestrator.retrieve_context")
def test_run_orchestrator_fallback_details(mock_retrieve_context):
    """Verify that run_orchestrator programmatically appends deep-dive details when LLM response lacks them."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_llm = MagicMock()
    mock_embed_model = MagicMock()
    
    mock_retrieve_context.return_value = {
        "confessional": [{"text": "Freely justified", "citation": "AC IV, 1"}],
        "scriptures": [{
            "citation": "Romans 3:28",
            "verse_id": 1,
            "primary_translation": "WEB",
            "cached_text": "Justified by faith",
        }]
    }
    
    # LLM returns ONLY summary
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Summary: We are justified by faith."
    mock_llm.invoke.return_value = mock_llm_response
    
    res = run_orchestrator(mock_chroma, mock_db, mock_llm, "Query", mock_embed_model, primary_translation="WEB")
    
    assert "Summary: We are justified by faith." in res
    assert "<details>" in res
    assert "<summary>Theological Depth</summary>" in res
    assert "AC IV, 1" in res
    assert "Romans 3:28 (WEB)" in res
    assert "Justified by faith" in res
    assert "Original Language Word Analysis" not in res
    assert "</details>" in res


@patch("pipeline.orchestrator.retrieve_context")
def test_run_orchestrator_multiple_citations(mock_retrieve_context):
    """Verify that run_orchestrator formats and includes multiple scripture citations and translations in the LLM prompt and deep-dive HTML."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_llm = MagicMock()
    mock_embed_model = MagicMock()
    
    # Configure mock retrieve_context to return multiple scriptures
    mock_retrieve_context.return_value = {
        "confessional": [{"text": "Freely justified", "citation": "AC IV, 1"}],
        "scriptures": [
            {
                "citation": "Romans 3:28",
                "verse_id": 1,
                "primary_translation": "WEB",
                "cached_text": "Justified by faith",
            },
            {
                "citation": "Ephesians 2:8",
                "verse_id": 2,
                "primary_translation": "WEB",
                "cached_text": "By grace you have been saved through faith",
            }
        ]
    }
    
    # LLM returns ONLY summary, triggering programmatic fallback deep-dive details appending
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Summary: We are saved and justified by faith through grace."
    mock_llm.invoke.return_value = mock_llm_response
    
    res = run_orchestrator(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        llm=mock_llm,
        query="Faith and grace query",
        embed_model=mock_embed_model,
        primary_translation="WEB"
    )
    
    # Verify that the LLM call prompt contained both scriptures and translations
    mock_llm.invoke.assert_called_once()
    messages = mock_llm.invoke.call_args[0][0]
    prompt_content = messages[0].content
    
    assert "Romans 3:28" in prompt_content
    assert "Justified by faith" in prompt_content
    assert "Ephesians 2:8" in prompt_content
    assert "By grace you have been saved through faith" in prompt_content
    
    # Verify programmatic deep-dive details output contains both scriptures in selected translation
    assert "Summary: We are saved and justified by faith through grace." in res
    assert "<details>" in res
    assert "<summary>Theological Depth</summary>" in res
    assert "AC IV, 1" in res
    
    assert "Romans 3:28 (WEB)" in res
    assert "Justified by faith" in res
    
    assert "Ephesians 2:8 (WEB)" in res
    assert "By grace you have been saved through faith" in res
    assert "Original Language Word Analysis" not in res
    assert "</details>" in res

