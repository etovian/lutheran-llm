from pipeline.guardrails import detect_pastoral_crisis, get_redirection_response

def test_detect_pastoral_crisis():
    """Verify that detect_pastoral_crisis returns True for crisis queries and False for normal queries."""
    assert detect_pastoral_crisis("I want to end my life") is True
    assert detect_pastoral_crisis("I feel so guilty I don't think God can save me") is True
    assert detect_pastoral_crisis("What is the Augsburg Confession?") is False
    assert detect_pastoral_crisis("How do Lutherans view baptism?") is False

def test_get_redirection_response():
    """Verify that get_redirection_response returns the standardized Gospel comfort and pastoral care direction."""
    response = get_redirection_response()
    assert "Gospel assurance" in response or "absolution" in response
    assert "pastor" in response or "absolution" in response

def test_detect_pastoral_crisis_null_and_empty():
    """Verify that None or empty queries do not crash and return False."""
    assert detect_pastoral_crisis("") is False
    assert detect_pastoral_crisis(None) is False

def test_detect_pastoral_crisis_theological_context():
    """Verify that academic or historical theological queries do not trigger false positives."""
    assert detect_pastoral_crisis("What did Martin Luther write about guilt?") is False
    assert detect_pastoral_crisis("How does the Book of Concord define despair?") is False
