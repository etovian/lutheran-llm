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


def test_parse_hebrew_words_with_strongs():
    """Verify that Hebrew words with 'H' Strong's numbers are correctly parsed."""
    raw_verse = "בְּרֵאשִׁ֖ית[H7225] בָּרָ֣א[H1254]"
    tokens = parse_original_word_tokens(raw_verse, verse_id=2)
    assert len(tokens) == 2
    assert tokens[0]["word_text"] == "בְּרֵאשִׁ֖ית"
    assert tokens[0]["strongs_number"] == "H7225"


def test_parse_bible_punctuation_handling():
    """Verify that attached punctuation is stripped from the word text."""
    raw_verse = "ἀρχῇ,[G746] λόγος.[G3056]"
    tokens = parse_original_word_tokens(raw_verse, verse_id=3)
    assert tokens[0]["word_text"] == "ἀρχῇ"
    assert tokens[1]["word_text"] == "λόγος"


def test_parse_bible_ignores_untagged_words():
    """Verify that words without Strong's numbers are ignored."""
    raw_verse = "Ἐν[G1722] ὁ λόγος"
    tokens = parse_original_word_tokens(raw_verse, verse_id=4)
    assert len(tokens) == 1
    assert tokens[0]["word_text"] == "Ἐν"
