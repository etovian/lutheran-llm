from unittest.mock import MagicMock, patch
import pytest
from ingestion.ingest_all import (
    clear_tables,
    seed_books,
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
    calls = [call[0][0].text.lower() for call in mock_conn.execute.call_args_list]
    
    deleted_tables = []
    for c in calls:
        for t in ["verse_translation", "verse", "book"]:
            if t in c:
                deleted_tables.append(t)
                break
                
    # Verify verse_translation deleted before verse
    assert deleted_tables.index("verse_translation") < deleted_tables.index("verse")
    # Verify verse deleted before book
    assert deleted_tables.index("verse") < deleted_tables.index("book")

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
    
    addr_to_id = parse_usfx_xml(xml_content, mock_conn)
    
    assert addr_to_id == {
        "GEN_1_1": 1,
        "GEN_1_2": 2,
        "ROM_3_28": 3
    }
    
    verse_inserts = [call for call in mock_conn.execute.call_args_list if "insert into verse " in call[0][0].text.lower()]
    assert len(verse_inserts) == 1
    assert len(verse_inserts[0][0][1]) == 3
    
    tx_inserts = [call for call in mock_conn.execute.call_args_list if "insert into verse_translation " in call[0][0].text.lower()]
    assert len(tx_inserts) == 1
    assert len(tx_inserts[0][0][1]) == 3
    
    word_inserts = [call for call in mock_conn.execute.call_args_list if "insert into original_word " in call[0][0].text.lower()]
    assert len(word_inserts) == 0

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
    
    inserts = [call for call in mock_conn.execute.call_args_list if "insert into verse_translation " in call[0][0].text.lower()]
    assert len(inserts) == 2  # 1 call for KJV, 1 call for MKJV
    
    inserted_params = []
    for call in inserts:
        params = call[0][1]
        if isinstance(params, list):
            inserted_params.extend(params)
        else:
            inserted_params.append(params)
            
    assert {"verse_id": 100, "version_code": "KJV", "text": "KJV Gen 1:1 text"} in inserted_params
    assert {"verse_id": 101, "version_code": "KJV", "text": "KJV Gen 1:2 text"} in inserted_params
    assert {"verse_id": 100, "version_code": "MKJV", "text": "MKJV Gen 1:1 text"} in inserted_params

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


@patch("ingestion.ingest_all.Settings")
@patch("ingestion.ingest_all.get_engine")
@patch("ingestion.ingest_all.parse_usfx_xml")
@patch("ingestion.ingest_all.parse_json_translations")
@patch("ingestion.ingest_all.parse_boc_markdown")
@patch("ingestion.ingest_all.chromadb.PersistentClient")
@patch("ingestion.ingest_all.VectorIndexer")
def test_run_ingest_all(
    mock_indexer_cls,
    mock_chroma_client_cls,
    mock_parse_boc,
    mock_parse_json,
    mock_parse_usfx,
    mock_get_engine,
    mock_settings_cls
):
    """Verify run_ingest_all coordinates database seeding, file parsing, and ChromaDB indexing."""
    mock_settings = MagicMock()
    mock_settings.primary_search_version = "WEB"
    mock_settings.chroma_db_path = "./test_chroma"
    mock_settings_cls.return_value = mock_settings
    
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_conn = mock_engine.begin.return_value.__enter__.return_value
    
    mock_parse_usfx.return_value = {"GEN_1_1": 1}
    mock_parse_boc.return_value = [
        {
            "text": "AC chunk text",
            "book": "Augsburg Confession",
            "article_id": "AC_I",
            "paragraph_number": 1,
            "citation": "AC I, 1"
        }
    ]
    
    mock_nt_conn = mock_engine.connect.return_value.__enter__.return_value
    mock_rows = MagicMock()
    mock_rows.mappings.return_value.all.return_value = [
        {
            "verse_id": 1,
            "address_code": "MAT_1_1",
            "chapter": 1,
            "verse_number": 1,
            "book_name": "Matthew",
            "text": "The book of the generation of Jesus Christ"
        }
    ]
    mock_nt_conn.execute.return_value = mock_rows
    
    mock_indexer = MagicMock()
    mock_indexer_cls.return_value = mock_indexer
    
    mock_chroma = MagicMock()
    mock_chroma_client_cls.return_value = mock_chroma
    
    run_ingest_all()
    
    assert mock_engine.begin.call_count == 1
    mock_parse_usfx.assert_called_once()
    mock_parse_json.assert_called_once()
    mock_parse_boc.assert_called_once()
    
    assert mock_chroma.delete_collection.call_count == 2
    assert mock_chroma.create_collection.call_count == 2
    
    mock_indexer.index_confessional_batch.assert_called_once_with(mock_parse_boc.return_value)
    mock_indexer.index_biblical_batch.assert_called_once()


