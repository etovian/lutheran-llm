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
                timeout=5
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
