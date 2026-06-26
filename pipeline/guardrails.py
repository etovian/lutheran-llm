"""
Guardrails module to detect pastoral crises and provide redirection responses.
"""
import re

CRISIS_PATTERNS = [
    r"\b(suicide|suicidal)\b",
    r"\b(kill myself|end my life|harm myself)\b",
    r"\b(unpardonable sin|cannot be saved|damned)\b",
    r"\b(feel|feeling)\s+(?:\w+\s+){0,2}(guilty|despair)\b",
    r"\bmy\s+guilt\b",
    r"\bi\s+despair\b"
]

def detect_pastoral_crisis(query: str) -> bool:
    """
    Perform a case-insensitive regex scan of the query to check for patterns indicating
    a personal pastoral crisis or spiritual distress, preventing false positives on 
    academic queries.
    
    Args:
        query (str): The user input query.
        
    Returns:
        bool: True if a crisis pattern is detected, False otherwise.
    """
    if not query:
        return False
        
    normalized_query = query.lower()
    return any(re.search(pattern, normalized_query) for pattern in CRISIS_PATTERNS)

def get_redirection_response() -> str:
    """
    Return the standardized comforting Gospel assurance and pastoral care redirection message.
    
    Returns:
        str: The redirection text.
    """
    return (
        "Be comforted by the Gospel assurance that Christ has paid for all your sins on the cross. "
        "Because this is a deeply personal or spiritual concern, please reach out to a local "
        "confessional Lutheran pastor who can deliver God's personal comfort, absolution, and guidance to you."
    )
