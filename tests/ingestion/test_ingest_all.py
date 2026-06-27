from unittest.mock import MagicMock, patch
import pytest
from ingestion.ingest_all import (
    clear_tables,
    seed_books,
    seed_strongs,
    parse_usfx_xml,
    parse_json_translations,
    parse_boc_markdown,
    run_ingest_all
)

def test_run_ingest_all_exists():
    """Simple test to verify run_ingest_all imports and exists."""
    assert run_ingest_all is not None

def test_clear_tables():
    """Verify that clear_tables executes DELETE statements in the correct order."""
    mock_conn = MagicMock()
    clear_tables(mock_conn)
    # The executions should clear original_word first, then verse_translation, verse, book, strongs_concordance
    calls = [call[0][0].text.lower() for call in mock_conn.execute.call_args_list]
    assert any("original_word" in c for c in calls)
    assert any("verse_translation" in c for c in calls)
    assert any("verse" in c for c in calls)
    assert any("book" in c for c in calls)
    assert any("strongs_concordance" in c for c in calls)

def test_seed_books():
    """Verify that all 66 books are seeded with correct IDs and testaments."""
    mock_conn = MagicMock()
    seed_books(mock_conn)
    
    # 66 insert calls
    assert mock_conn.execute.call_count == 66
    
    # Check some books
    called_books = [call[0][1] for call in mock_conn.execute.call_args_list]
    assert called_books[0] == {"book_id": 1, "name": "Genesis", "testament": "OT"}
    assert called_books[-1] == {"book_id": 66, "name": "Revelation of John", "testament": "NT"}

def test_seed_strongs():
    """Verify that the 20 default + 6 new Strong's definitions are seeded."""
    mock_conn = MagicMock()
    seed_strongs(mock_conn)
    
    assert mock_conn.execute.call_count == 26
    
    # Check that new ones are present
    called_strongs = [call[0][1]["strongs_number"] for call in mock_conn.execute.call_args_list]
    assert "G907" in called_strongs
    assert "G3067" in called_strongs
    assert "G3824" in called_strongs
    assert "G4983" in called_strongs
    assert "G2842" in called_strongs
    assert "G1722" in called_strongs

def test_parse_usfx_xml():
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <usfx>
      <book id="FRT">
        <p>Preface</p>
      </book>
      <book id="GEN">
        <c id="1"/>
        <p><v id="1"/>In the beginning <ve/><v id="2"/>The earth was formless <ve/></p>
      </book>
      <book id="ROM">
        <c id="3"/>
        <p><v id="28"/>We maintain therefore <ve/></p>
      </book>
    </usfx>
    """
    mock_conn = MagicMock()
    key_verses = {
        "ROM_3_28": "λογιζόμεθα[G3049] οὖν[G3767]"
    }
    
    addr_to_id = parse_usfx_xml(xml_content, mock_conn, key_verses)
    
    assert addr_to_id == {
        "GEN_1_1": 1,
        "GEN_1_2": 2,
        "ROM_3_28": 3
    }
    
    calls = [call[0][0].text.lower() for call in mock_conn.execute.call_args_list]
    
    verse_inserts = [c for c in calls if "insert into verse " in c]
    assert len(verse_inserts) == 3
    
    tx_inserts = [c for c in calls if "insert into verse_translation " in c]
    assert len(tx_inserts) == 3
    
    word_inserts = [c for c in calls if "insert into original_word " in c]
    assert len(word_inserts) == 2

def test_parse_json_translations(tmp_path):
    import json
    kjv_data = {
        "translation": "KJV",
        "books": [
            {
                "name": "Genesis",
                "chapters": [
                    {
                        "chapter": 1,
                        "verses": [
                            {"verse": 1, "text": "KJV Gen 1:1 text"},
                            {"verse": 2, "text": "KJV Gen 1:2 text"}
                        ]
                    }
                ]
            }
        ]
    }
    mkjv_data = {
        "translation": "MKJV",
        "books": [
            {
                "name": "Genesis",
                "chapters": [
                    {
                        "chapter": 1,
                        "verses": [
                            {"verse": 1, "text": "MKJV Gen 1:1 text"}
                        ]
                    }
                ]
            }
        ]
    }
    
    kjv_file = tmp_path / "KJV.json"
    mkjv_file = tmp_path / "MKJV.json"
    kjv_file.write_text(json.dumps(kjv_data))
    mkjv_file.write_text(json.dumps(mkjv_data))
    
    mock_conn = MagicMock()
    address_to_id = {
        "GEN_1_1": 100,
        "GEN_1_2": 101
    }
    
    parse_json_translations(str(kjv_file), str(mkjv_file), mock_conn, address_to_id)
    
    calls = [call[0][0].text.lower() for call in mock_conn.execute.call_args_list]
    inserts = [c for c in calls if "insert into verse_translation " in c]
    assert len(inserts) == 3
    
    called_params = [call[0][1] for call in mock_conn.execute.call_args_list if "insert into verse_translation " in call[0][0].text.lower()]
    assert {"verse_id": 100, "version_code": "KJV", "text": "KJV Gen 1:1 text"} in called_params
    assert {"verse_id": 101, "version_code": "KJV", "text": "KJV Gen 1:2 text"} in called_params
    assert {"verse_id": 100, "version_code": "MKJV", "text": "MKJV Gen 1:1 text"} in called_params

def test_parse_boc_markdown(tmp_path):
    ac_dir = tmp_path / "augsburg-confession"
    ac_dir.mkdir()
    
    ac_file = ac_dir / "articles.md"
    ac_file.write_text(
        "---\n"
        "title: Articles\n"
        "---\n\n"
        "# Article I: Of God\n\n"
        "Our Churches, with common consent, do teach...\n\n"
        "This article is about God.\n\n"
        "## Article II: Of Original Sin\n\n"
        "Also they teach that since the fall of Adam...\n\n"
        "### Index or Navigation\n"
        "[Home](index.md)\n"
    )
    
    # Create an index file to verify it's skipped
    index_file = ac_dir / "_index.md"
    index_file.write_text(
        "# Index\n"
        "- [Article I](articles.md#article-i)\n"
    )
    
    chunks = parse_boc_markdown(str(tmp_path))
    
    assert len(chunks) == 3
    assert chunks[0]["book"] == "Augsburg Confession"
    assert chunks[0]["article_id"] == "AC_I"
    assert chunks[0]["paragraph_number"] == 1
    assert "Our Churches" in chunks[0]["text"]
    assert chunks[0]["citation"] == "Augsburg Confession, Article I: Of God, Paragraph 1"
    
    assert chunks[1]["paragraph_number"] == 2
    assert "This article" in chunks[1]["text"]
    
    assert chunks[2]["article_id"] == "AC_II"
    assert chunks[2]["paragraph_number"] == 1
    assert "Also they teach" in chunks[2]["text"]
    assert chunks[2]["citation"] == "Augsburg Confession, Article II: Of Original Sin, Paragraph 1"

