import logging
from sqlalchemy import create_engine, text
from config.settings import Settings

logger = logging.getLogger(__name__)

def get_engine(settings=None):
    """Create and return a SQLAlchemy engine from database URL settings."""
    if settings is None:
        settings = Settings()
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
    except Exception as e:
        logger.error("Database connection check failed: %s", e, exc_info=True)
        return False
