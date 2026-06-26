import re

def parse_original_word_tokens(raw_verse_text: str, verse_id: int) -> list[dict]:
    """
    Tokenize a raw verse string containing original language words annotated with Strong's Concordance tags.
    
    Args:
        raw_verse_text (str): The raw verse text containing original words and Strong's numbers (e.g. "Ἐν[G1722]").
        verse_id (int): The unique identifier of the verse.
        
    Returns:
        list[dict]: A list of dicts representing original language word tokens with Strong's numbers and positional indexes.
    """
    pattern = re.compile(r"([^\s\[]+)\[([GH]\d+)\]")
    tokens = []
    
    matches = pattern.finditer(raw_verse_text)
    for idx, match in enumerate(matches):
        word_text = match.group(1)
        strongs_num = match.group(2)
        
        tokens.append({
            "verse_id": verse_id,
            "word_index": idx,
            "word_text": word_text,
            "lemma": word_text,
            "strongs_number": strongs_num
        })
        
    return tokens
