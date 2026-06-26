import requests
from langchain_core.messages import AIMessage

class OllamaChatModel:
    """
    A custom LangChain-compatible wrapper for local Ollama chat models.
    """
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

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
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            res_json = response.json()
            content = res_json.get("message", {}).get("content", "")
            return AIMessage(content=content)
        except Exception as e:
            raise RuntimeError(f"Ollama invocation failed: {e}")
