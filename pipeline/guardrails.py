"""
Guardrails module to detect pastoral crises and provide redirection responses.
"""

CRISIS_KEYWORDS = [
    "guilt", 
    "unpardonable", 
    "kill myself", 
    "end my life", 
    "suicide", 
    "despair", 
    "cannot be saved"
]

def detect_pastoral_crisis(query: str) -> bool:
    """
    Perform a case-insensitive scan of the query to check for keywords indicating
    a personal pastoral crisis or spiritual distress.
    
    Args:
        query (str): The user input query.
        
    Returns:
        bool: True if a crisis keyword is detected, False otherwise.
    """
    normalized_query = query.lower()
    return any(keyword in normalized_query for keyword in CRISIS_KEYWORDS)

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
