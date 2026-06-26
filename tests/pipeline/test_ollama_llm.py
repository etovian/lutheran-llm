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
            "stream": False,
            "options": {
                "num_predict": 150,
                "temperature": 0.0,
                "num_ctx": 1024
            }
        },
        timeout=300
    )

@patch("requests.post")
def test_ollama_chat_model_invoke_failure(mock_post):
    mock_post.side_effect = Exception("Connection refused")
    
    model = OllamaChatModel()
    with pytest.raises(RuntimeError, match="Ollama invocation failed: Connection refused"):
        model.invoke([HumanMessage(content="Hello")])


@patch("os.path.isfile")
@patch("os.environ.get")
def test_get_ollama_executable_path(mock_env, mock_isfile):
    mock_env.return_value = "C:\\Users\\Test"
    mock_isfile.return_value = True
    
    from pipeline.ollama_llm import get_ollama_executable_path
    assert get_ollama_executable_path() == "C:\\Users\\Test\\Programs\\Ollama\\ollama.exe"


@patch("requests.get")
@patch("subprocess.Popen")
def test_start_ollama_server_already_running(mock_popen, mock_get):
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_get.return_value = mock_res
    
    from pipeline.ollama_llm import start_ollama_server
    assert start_ollama_server() is True
    mock_popen.assert_not_called()


@patch("requests.get")
@patch("requests.post")
def test_ensure_model_loaded_success(mock_post, mock_get):
    mock_res = MagicMock()
    mock_res.json.return_value = {"models": [{"name": "llama3:latest"}]}
    mock_get.return_value = mock_res
    
    from pipeline.ollama_llm import ensure_model_loaded
    success, msg = ensure_model_loaded(model_name="llama3")
    assert success is True
    assert "ready" in msg
    mock_post.assert_not_called()
