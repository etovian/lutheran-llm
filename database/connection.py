from sqlalchemy import create_engine, text
from config.settings import Settings

settings = Settings()

def get_engine():
    """Create and return a SQLAlchemy engine from database URL settings."""
    return create_engine(settings.database_url)

def check_connection(engine):
    """
    Attempt to connect to the engine and run a simple query.
    Returns True if successful, and False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception:
        return False
