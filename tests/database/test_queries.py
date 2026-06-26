from unittest.mock import MagicMock
from database.queries import fetch_parallel_verses_and_lexicon

def test_fetch_parallel_verses_and_lexicon():
    """Verify that parallel translations and Strong's lexicon are queried and returned correctly."""
    mock_engine = MagicMock()
    mock_connection = mock_engine.connect.return_value.__enter__.return_value
    
    # Configure mock SQL execution returns
    # First call: verse translations query
    # Second call: original words and concordance join query
    mock_result_tx = MagicMock()
    mock_result_tx.mappings.return_value.all.return_value = [
        {"version_code": "WEB", "text": "We maintain therefore that a man is justified..."},
        {"version_code": "KJV", "text": "Therefore we conclude that a man is justified..."}
    ]
    
    mock_result_lex = MagicMock()
    mock_result_lex.mappings.return_value.all.return_value = [
        {
            "word_text": "λογιζόμεθα", 
            "lemma": "λογίζομαι",
            "strongs_number": "G3049", 
            "pronunciation": "log-id'-zom-ahee",
            "definition": "to reckon, calculate",
            "derivation": "middle voice of..."
        }
    ]
    
    mock_connection.execute.side_effect = [mock_result_tx, mock_result_lex]
    
    res = fetch_parallel_verses_and_lexicon(mock_engine, verse_id=1)
    
    assert "translations" in res
    assert res["translations"]["WEB"] == "We maintain therefore that a man is justified..."
    assert res["translations"]["KJV"] == "Therefore we conclude that a man is justified..."
    
    assert "lexicon" in res
    assert len(res["lexicon"]) == 1
    assert res["lexicon"][0]["word_text"] == "λογιζόμεθα"
    assert res["lexicon"][0]["lemma"] == "λογίζομαι"
    assert res["lexicon"][0]["strongs_number"] == "G3049"
    assert res["lexicon"][0]["definition"] == "to reckon, calculate"
