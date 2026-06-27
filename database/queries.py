import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

def fetch_parallel_translations(engine, verse_id: int) -> dict:
    """
    Fetch parallel verse translations from the relational database for a given verse_id.
    
    Args:
        engine: SQLAlchemy engine instance.
        verse_id (int): Unique identifier of the verse.
        
    Returns:
        dict: A dictionary mapping version_code to text, e.g. {"WEB": "...", "KJV": "...", "MKJV": "..."}
    """
    translations = {}
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
    except Exception as e:
        logger.error("Failed to fetch parallel translations for verse_id %d: %s", verse_id, e, exc_info=True)
        raise
        
    return translations


def fetch_single_translation(engine, verse_id: int, version_code: str) -> str:
    """
    Fetch a single verse translation from the relational database.

    Args:
        engine: SQLAlchemy engine instance.
        verse_id (int): Unique identifier of the verse.
        version_code (str): Translation version code, e.g. "WEB", "KJV", "MKJV".

    Returns:
        str: The verse text, or empty string if not found.
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT text FROM verse_translation
                    WHERE verse_id = :verse_id AND version_code = :version_code
                """),
                {"verse_id": verse_id, "version_code": version_code}
            ).mappings().first()
            return row["text"] if row else ""
    except Exception as e:
        logger.error(
            "Failed to fetch translation for verse_id %d version %s: %s",
            verse_id, version_code, e, exc_info=True
        )
        return ""
