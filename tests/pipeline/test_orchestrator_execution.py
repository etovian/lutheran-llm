from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.orchestrator import run_orchestrator, format_deep_dive_details, format_llm_context

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


def test_format_deep_dive_details_collapsible():
    """Verify that format_deep_dive_details wraps confessional chunks in collapsible details tags."""
    ctx = {
        "confessional": [{"text": "Freely justified by grace", "citation": "Apology IV, 1"}],
        "scriptures": []
    }
    html_output = format_deep_dive_details(ctx, primary_translation="WEB", db_engine=None)
    assert '<details class="boc-detail"' in html_output
    assert '<summary style="font-weight: 500; font-size: 0.95rem; color: #E2E8F0; cursor: pointer;">Apology IV, 1</summary>' in html_output
    assert '"Freely justified by grace"' in html_output


def test_format_llm_context_adds_reference_labels():
    ctx = {
        "confessional": [{"text": "Justified by faith", "citation": "AC IV"}],
        "scriptures": [{"citation": "Romans 3:28", "primary_translation": "WEB", "cached_text": "Justified by faith"}]
    }
    formatted = format_llm_context(ctx)
    assert "[Ref-1] Source: AC IV" in formatted
    assert "[Ref-2] Citation: Romans 3:28" in formatted


@patch("pipeline.orchestrator.retrieve_context")
def test_run_orchestrator_filters_citations(mock_retrieve_context):
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_llm = MagicMock()
    mock_embed_model = MagicMock()
    
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
                "cached_text": "Saved by grace",
            }
        ]
    }
    
    # LLM returns text citing only the scripture Romans 3:28 [Ref-2]
    mock_llm_response = MagicMock()
    mock_llm_response.content = "We are justified by faith. <citations>[Ref-2]</citations>"
    mock_llm.invoke.return_value = mock_llm_response
    
    res = run_orchestrator(mock_chroma, mock_db, mock_llm, "Query", mock_embed_model)
    
    # The output should NOT contain <citations> tags or Ref-2 in text
    assert "<citations>" not in res
    assert "We are justified by faith." in res
    # The deep-dive section should have Romans 3:28 (cited) but NOT Ephesians 2:8 (not cited)
    assert "Romans 3:28" in res
    assert "Ephesians 2:8" not in res
    # Confessional is also not cited, so it should not be present
    assert "AC IV, 1" not in res


def test_format_boc_text_with_table():
    from pipeline.orchestrator import format_boc_text
    
    text = (
        "| Header 1 | Header 2 |\n"
        "|---|---|\n"
        "| Cell 1 | Cell 2 |"
    )
    res = format_boc_text(text)
    assert '<table class="boc-table">' in res
    assert '<th>Header 1</th>' in res
    assert '<td>Cell 1</td>' in res





