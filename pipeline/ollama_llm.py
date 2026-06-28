import requests
from langchain_core.messages import AIMessage

class OllamaChatModel:
    """
    A custom LangChain-compatible wrapper for local Ollama chat models.
    """
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434", options: dict = None):
        self.model_name = model_name
        self.base_url = base_url
        self.options = options or {
            "num_predict": 512,
            "temperature": 0.0,
            "num_ctx": 1024
        }

    def invoke(self, messages: list) -> AIMessage:
        formatted_messages = []
        for msg in messages:
            # Determine role from message type if it is a LangChain message object
            if hasattr(msg, "type"):
                role = "system" if msg.type == "system" else "user"
            else:
                role = "user"
            
            content = msg.content if hasattr(msg, "content") else str(msg)
            formatted_messages.append({
                "role": role,
                "content": content
            })

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": formatted_messages,
                    "stream": False,
                    "options": self.options
                },
                timeout=600
            )
            response.raise_for_status()
            res_json = response.json()
            content = res_json.get("message", {}).get("content", "")
            return AIMessage(content=content)
        except Exception as e:
            raise RuntimeError(f"Ollama invocation failed: {e}")


class SimulatedChatModel:
    """
    A simulated ChatModel that generates a realistic RAG response based on 
    the actual database/vector context when Ollama is offline.
    """
    def invoke(self, messages: list) -> AIMessage:
        system_msg = next((msg.content for msg in messages if hasattr(msg, "type") and msg.type == "system"), "")
        
        # Parse the structured context from the system message
        confessional_lines = []
        scripture_lines = []
        lexicon_lines = []
        
        in_conf = False
        in_scrip = False
        
        for line in system_msg.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "CONFESSIONAL CONTEXT" in line:
                in_conf = True
                in_scrip = False
                continue
            elif "SCRIPTURE CONTEXT" in line:
                in_conf = False
                in_scrip = True
                continue
            
            if in_conf:
                if line.startswith("Source:") or line.startswith("Content:"):
                    confessional_lines.append(line)
            elif in_scrip:
                if line.startswith("[") or line.startswith("- Word:") or line.startswith("-"):
                    if line.startswith("- Word:") or (line.startswith("-") and "strongs" in line.lower()):
                        lexicon_lines.append(line)
                    else:
                        scripture_lines.append(line)

        # Generate summary
        summary = (
            "We are justified freely by God's grace through faith for Christ's sake, "
            "apart from works of the law. Christ has fully satisfied God's justice by "
            "His death on the cross, and this righteousness is imputed to all who believe."
        )
        
        # Format theological deep dive HTML
        depth = ["<details>", "<summary>Theological Depth</summary>", "<h4>Book of Concord Citations</h4>"]
        
        current_citation = "Book of Concord"
        for line in confessional_lines:
            if line.startswith("Source:"):
                current_citation = line.replace("Source:", "").strip()
            elif line.startswith("Content:"):
                content_text = line.replace("Content:", "").strip()
                depth.append(f"<p><strong>{current_citation}</strong>: <em>\"{content_text}\"</em></p>")
                
        if scripture_lines:
            depth.append("<h4>Parallel Bible Translations</h4><ul>")
            for line in scripture_lines:
                depth.append(f"<li>{line}</li>")
            depth.append("</ul>")
            
        if lexicon_lines:
            depth.append("<h4>Original Language Word Analysis</h4><ul>")
            for line in lexicon_lines:
                # Remove starting dash
                display_line = line.lstrip("- ").strip()
                depth.append(f"<li>{display_line}</li>")
            depth.append("</ul>")
            
        depth.append("</details>")
        
        return AIMessage(content=f"{summary}\n\n" + "\n".join(depth))


import os
import subprocess
import time
import logging

logger = logging.getLogger(__name__)

def get_ollama_executable_path() -> str:
    """Resolve the absolute path of the Ollama executable on Windows or fallback to command."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        default_path = os.path.join(local_appdata, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(default_path):
            return default_path
    return "ollama"

def start_ollama_server(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama server is running. If not, start it in the background.
    Returns True if running (or successfully started), False otherwise.
    """
    try:
        res = requests.get(base_url, timeout=0.5)
        if res.status_code == 200:
            logger.info("Ollama server is already running.")
            return True
    except Exception:
        pass

    executable = get_ollama_executable_path()
    logger.info(f"Starting Ollama server using executable: {executable}")
    
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = 0x08000000 # CREATE_NO_WINDOW
            
        subprocess.Popen(
            [executable, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True
        )
        
        # Poll server until it responds (up to 10 seconds)
        for _ in range(20):
            time.sleep(0.5)
            try:
                res = requests.get(base_url, timeout=0.5)
                if res.status_code == 200:
                    logger.info("Ollama server started successfully.")
                    return True
            except Exception:
                pass
        logger.error("Ollama server process triggered but did not respond on port 11434.")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama server process: {e}")
        return False

def ensure_model_loaded(base_url: str = "http://localhost:11434", model_name: str = "llama3") -> tuple[bool, str]:
    """
    Query Ollama server to verify if model_name is present.
    If missing, trigger a pull request.
    Returns (success_status, status_message).
    """
    try:
        res = requests.get(f"{base_url}/api/tags", timeout=2)
        res.raise_for_status()
        models = res.json().get("models", [])
        
        model_names = [m.get("name") for m in models]
        model_base_names = [name.split(":")[0] for name in model_names if name]
        
        if model_name in model_names or model_name in model_base_names:
            return True, f"Model '{model_name}' is already loaded and ready."
            
        logger.info(f"Model '{model_name}' is missing. Initiating pull request...")
        pull_res = requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name, "stream": False},
            timeout=600 # 10 minutes timeout for model download
        )
        pull_res.raise_for_status()
        logger.info(f"Successfully pulled model '{model_name}'.")
        return True, f"Successfully pulled model '{model_name}'."
    except Exception as e:
        logger.error(f"Failed to verify or pull model '{model_name}': {e}")
        return False, str(e)


class GroqChatModel:
    """
    A custom LangChain-compatible wrapper for the Groq serverless API.
    """
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192", temperature: float = 0.0, max_tokens: int = 512):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages: list) -> AIMessage:
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "type"):
                role = "system" if msg.type == "system" else "user"
            else:
                role = "user"
            content = msg.content if hasattr(msg, "content") else str(msg)
            formatted_messages.append({
                "role": role,
                "content": content
            })

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": formatted_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                timeout=60
            )
            if response.status_code != 200:
                try:
                    err_msg = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                raise RuntimeError(err_msg)
            response.raise_for_status()
            res_json = response.json()
            choices = res_json.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            return AIMessage(content=content)
        except Exception as e:
            raise RuntimeError(f"Groq API invocation failed: {e}")

