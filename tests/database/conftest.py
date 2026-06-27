import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def db_engine():
    """Fixture to create an in-memory SQLite database populated with the schema."""
    engine = create_engine("sqlite:///:memory:")
    with open("database/schema.sql", "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            clean_stmt = statement.strip()
            if clean_stmt:
                conn.execute(text(clean_stmt))
    return engine
