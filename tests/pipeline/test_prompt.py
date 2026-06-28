from pipeline.prompt import SYSTEM_PROMPT

def test_system_prompt_hermeneutical_lens():
    """Verify that the SYSTEM_PROMPT contains the hermeneutical circle/lens instruction."""
    assert "hermeneutical lens" in SYSTEM_PROMPT
    assert "norma normans" in SYSTEM_PROMPT
    assert "norma normata" in SYSTEM_PROMPT
    assert "Law/Gospel distinction" in SYSTEM_PROMPT
