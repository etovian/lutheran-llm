from ingestion.parse_bible import parse_original_word_tokens

def test_parse_greek_words_with_strongs():
    """Verify that a raw verse string is correctly tokenized into original language words and Strong's numbers."""
    raw_verse = "Ἐν[G1722] ἀρχῇ[G746] ἦν[G2258]"
    tokens = parse_original_word_tokens(raw_verse, verse_id=1)
    
    assert len(tokens) == 3
    
    assert tokens[0]["verse_id"] == 1
    assert tokens[0]["word_index"] == 0
    assert tokens[0]["word_text"] == "Ἐν"
    assert tokens[0]["lemma"] == "Ἐν"
    assert tokens[0]["strongs_number"] == "G1722"
    
    assert tokens[1]["word_index"] == 1
    assert tokens[1]["word_text"] == "ἀρχῇ"
    assert tokens[1]["strongs_number"] == "G746"
    
    assert tokens[2]["word_index"] == 2
    assert tokens[2]["word_text"] == "ἦν"
    assert tokens[2]["strongs_number"] == "G2258"
