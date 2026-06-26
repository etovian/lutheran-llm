from unittest.mock import MagicMock, patch
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.orchestrator import run_orchestrator

@patch("pipeline.orchestrator.retrieve_context")
def test_run_orchestrator(mock_retrieve_context):
    """Verify that run_orchestrator retrieves context, formats prompt, and calls LLM correctly."""
    mock_chroma = MagicMock()
    mock_db = MagicMock()
    mock_llm = MagicMock()
    mock_embed_model = MagicMock()
    
    # Configure mock retrieve_context return values
    mock_retrieve_context.return_value = {
        "confessional": [{"text": "Freely justified", "citation": "AC IV, 1"}],
        "scripture": {
            "translations": {"WEB": "Justified by faith"},
            "lexicon": [{"word_text": "δικαιοῦσθαι", "lemma": "δικαιόω", "strongs_number": "G1344", "pronunciation": "dik-ah-yo'-o", "definition": "to justify"}],
        }
    }
    
    # Configure mock LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Summary: We are justified by faith.\n<details>\n<summary>Theological Depth</summary>\n..."
    mock_llm.invoke.return_value = mock_llm_response
    
    res = run_orchestrator(
        chroma_client=mock_chroma,
        db_engine=mock_db,
        llm=mock_llm,
        query="How are we justified?",
        embed_model=mock_embed_model
    )
    
    assert "Summary: We are justified by faith." in res
    
    mock_retrieve_context.assert_called_once_with(mock_chroma, mock_db, "How are we justified?", mock_embed_model)
    
    # Verify LLM invoke was called with SystemMessage (containing context) and HumanMessage
    mock_llm.invoke.assert_called_once()
    messages = mock_llm.invoke.call_args[0][0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Freely justified" in messages[0].content
    assert "δικαιοῦσθαι" in messages[0].content
    assert messages[1].content == "How are we justified?"
