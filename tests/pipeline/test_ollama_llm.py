from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pipeline.ollama_llm import OllamaChatModel

@patch("requests.post")
def test_ollama_chat_model_invoke_success(mock_post):
    # Configure mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "Comforting Lutheran guidance."
        }
    }
    mock_post.return_value = mock_response

    model = OllamaChatModel(model_name="llama3", base_url="http://localhost:11434")
    messages = [
        SystemMessage(content="You are a Lutheran assistant."),
        HumanMessage(content="What is grace?")
    ]
    
    result = model.invoke(messages)
    
    assert isinstance(result, AIMessage)
    assert result.content == "Comforting Lutheran guidance."
    
    mock_post.assert_called_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "messages": [
                {"role": "system", "content": "You are a Lutheran assistant."},
                {"role": "user", "content": "What is grace?"}
            ],
            "stream": False
        },
        timeout=30
    )

@patch("requests.post")
def test_ollama_chat_model_invoke_failure(mock_post):
    mock_post.side_effect = Exception("Connection refused")
    
    model = OllamaChatModel()
    with pytest.raises(RuntimeError, match="Ollama invocation failed: Connection refused"):
        model.invoke([HumanMessage(content="Hello")])
