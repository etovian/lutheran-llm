# Lexicon Removal & Dynamic Translation Rendering Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove Greek lexicon/Strong's tables and logic entirely from the codebase, default RAG retrieval count bounds to 2 Book of Concord passages and 10 Scripture passages, and implement dynamic single-translation UI rendering that updates instantly when changed in the sidebar.

**Architecture:** We will simplify the database schema and ingestion seeder by dropping the `original_word` and `strongs_concordance` tables. The retriever will output structured scripture objects without lexicon fields, and the Streamlit UI will store structured messages, dynamically generating the HTML collapsible block for the selected translation during page rendering.

**Tech Stack:** Python, PostgreSQL (SQLAlchemy), ChromaDB, Streamlit, Pytest

---

### Task 1: Database Schema Simplification

**Files:**
- Modify: `database/schema.sql`
- Modify: `tests/database/test_connection.py`

**Step 1: Write the failing test**
In `tests/database/test_connection.py:16-24`, update the assertion of expected database tables to remove `"strongs_concordance"` and `"original_word"`.
```python
def test_tables_created(db_engine):
    inspector = inspect(db_engine)
    tables = inspector.get_table_names()
    assert "book" in tables
    assert "verse" in tables
    assert "verse_translation" in tables
    assert "strongs_concordance" not in tables
    assert "original_word" not in tables
```

**Step 2: Run test to verify it fails**
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/database/test_connection.py`
Expected: FAIL (since schema.sql still creates those tables).

**Step 3: Write minimal implementation**
In `database/schema.sql`, delete the table creation SQL blocks for `strongs_concordance` and `original_word` (and any related foreign keys).

**Step 4: Run test to verify it passes**
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/database/test_connection.py`
Expected: PASS

**Step 5: Commit**
```bash
git add database/schema.sql tests/database/test_connection.py
git commit -m "refactor: remove lexicon tables from database schema"
```

---

### Task 2: Ingestion & Seeder Pipeline Refactoring

**Files:**
- Modify: `ingestion/ingest_all.py`
- Modify: `tests/ingestion/test_ingest_all.py`

**Step 1: Write the failing test**
In `tests/ingestion/test_ingest_all.py`, update `test_clear_tables` to remove delete assertions for `original_word` and `strongs_concordance`. Remove the `test_seed_strongs` unit test entirely.
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/ingestion/test_ingest_all.py`
Expected: FAIL (due to missing seeder expectations or imports).

**Step 2: Write minimal implementation**
In `ingestion/ingest_all.py`:
- Remove `STRONGS_CONCORDANCE` seeder lists and `seed_strongs(conn)` function.
- Remove `parse_original_word_tokens` helper.
- Update `clear_tables(conn)` to only execute deletions for `verse_translation` -> `verse` -> `book` tables in correct order.
- Remove the `original_verse` tagged string mapping and insertion logic from `parse_usfx_xml`.
- Update `run_ingest_all()` coordinates to bypass lexicon indexing.

**Step 3: Run test to verify it passes**
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/ingestion/test_ingest_all.py`
Expected: PASS

**Step 4: Commit**
```bash
git add ingestion/ingest_all.py tests/ingestion/test_ingest_all.py
git commit -m "refactor: remove lexicon parsing and seeding from ingestion script"
```

---

### Task 3: RAG Retrieval Tuning & Query Simplification

**Files:**
- Modify: `config/settings.py`
- Modify: `database/queries.py`
- Modify: `pipeline/orchestrator.py`
- Modify: `tests/config/test_settings.py`
- Modify: `tests/database/test_queries.py`
- Modify: `tests/pipeline/test_orchestrator_retrieval.py`

**Step 1: Write failing tests**
- In `tests/config/test_settings.py`, assert that `settings.rag_confessional_k == 2` and `settings.rag_biblical_k == 10`.
- In `tests/database/test_queries.py`, replace `test_fetch_parallel_verses_and_lexicon` with `test_fetch_parallel_translations` to verify it returns only translation strings mapping (no lexicon).
- In `tests/pipeline/test_orchestrail_retrieval.py`, update mock queries to check retrieved scriptures structure (excluding `lexicon`).
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/config/test_settings.py tests/database/test_queries.py tests/pipeline/test_orchestrator_retrieval.py`
Expected: FAIL

**Step 2: Write minimal implementation**
- In `config/settings.py`, update defaults to `rag_confessional_k: int = 2` and `rag_biblical_k: int = 10`.
- In `database/queries.py`, remove `fetch_parallel_verses_and_lexicon` and replace it with:
  ```python
  def fetch_parallel_translations(db_engine, verse_id: int) -> dict:
      # Query verse_translation and return dictionary mapping version_code to text
  ```
- In `pipeline/orchestrator.py`:
  - Update `retrieve_context` to call `fetch_parallel_translations` instead of lexicon query.
  - Return each scripture in the list under the format: `{"citation": "...", "translations": {"WEB": "...", "KJV": "...", "MKJV": "..."}}`.

**Step 3: Run tests to verify they pass**
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/config/test_settings.py tests/database/test_queries.py tests/pipeline/test_orchestrator_retrieval.py`
Expected: PASS

**Step 4: Commit**
```bash
git add config/settings.py database/queries.py pipeline/orchestrator.py tests/config/test_settings.py tests/database/test_queries.py tests/pipeline/test_orchestrator_retrieval.py
git commit -m "feat: default RAG context to confessional k=2 and biblical k=10, simplify retrieval queries"
```

---

### Task 4: Dynamic Re-Rendering UI Integration

**Files:**
- Modify: `pipeline/orchestrator.py`
- Modify: `ui/app.py`
- Modify: `tests/pipeline/test_orchestrator_execution.py`

**Step 1: Write failing tests**
Update `tests/pipeline/test_orchestrator_execution.py` to test:
- `format_deep_dive_details(retrieved_ctx, primary_translation)` formats Book of Concord citations and a single selected translation of the Scripture (no lexicon output).
- `run_orchestrator` returns the LLM response summary, appending collapsible HTML output rendered with the selected translation.
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/pipeline/test_orchestrator_execution.py`
Expected: FAIL

**Step 2: Write minimal implementation**
- In `pipeline/orchestrator.py`:
  - Refactor `format_deep_dive_details(retrieved_ctx: dict, primary_translation: str) -> str` to build the collapsible block using *only* Book of Concord citations (most relevant 2) and Scripture citations displaying the text in the requested `primary_translation`. Remove the Original Language Word Analysis section.
  - Refactor `run_orchestrator` to accept `primary_translation: str = "WEB"` and pass it to `format_deep_dive_details`.
- In `ui/app.py`:
  - Update `st.session_state.messages` list payload for the assistant. When a response is generated, store the response as a dictionary:
    ```python
    {"role": "assistant", "summary": response_summary, "retrieved_ctx": retrieved_ctx}
    ```
  - During the message rendering loop:
    - If `role == "user"`, call `st.markdown(content)`.
    - If `role == "assistant"`, call `st.markdown(summary)` followed by:
      ```python
      details_html = format_deep_dive_details(message["retrieved_ctx"], selected_version)
      st.markdown(details_html, unsafe_allow_html=True)
      ```

**Step 3: Run all tests to verify they pass**
Run: `$env:PYTHONPATH="."; .\.venv\Scripts\pytest`
Expected: PASS

**Step 4: Commit**
```bash
git add pipeline/orchestrator.py ui/app.py tests/pipeline/test_orchestrator_execution.py
git commit -m "feat: implement dynamic translation rendering in streamlit UI"
```
