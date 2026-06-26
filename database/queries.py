import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

def fetch_parallel_verses_and_lexicon(engine, verse_id: int) -> dict:
    """
    Fetch parallel verse translations and original word concordance definitions
    from the relational database for a given verse_id.
    
    Args:
        engine: SQLAlchemy engine instance.
        verse_id (int): Unique identifier of the verse.
        
    Returns:
        dict: A dictionary containing parallel 'translations' and 'lexicon' analysis.
    """
    translations = {}
    lexicon = []
    
    try:
        with engine.connect() as conn:
            tx_query = text("""
                SELECT version_code, text
                FROM verse_translation
                WHERE verse_id = :verse_id
            """)
            tx_rows = conn.execute(tx_query, {"verse_id": verse_id}).mappings().all()
            for row in tx_rows:
                translations[row["version_code"]] = row["text"]
                
            lex_query = text("""
                SELECT w.word_text, w.lemma, w.strongs_number, 
                       s.pronunciation, s.definition, s.derivation
                FROM original_word w
                JOIN strongs_concordance s ON w.strongs_number = s.strongs_number
                WHERE w.verse_id = :verse_id
                ORDER BY w.word_index ASC
            """)
            lex_rows = conn.execute(lex_query, {"verse_id": verse_id}).mappings().all()
            lexicon = [dict(row) for row in lex_rows]
            
    except Exception as e:
        logger.error("Failed to fetch parallel verses and lexicon for verse_id %d: %s", verse_id, e, exc_info=True)
        raise
        
    return {
        "translations": translations,
        "lexicon": lexicon
    }
