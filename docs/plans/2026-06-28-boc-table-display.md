# Book of Concord Tabular Citations Display Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Detect markdown tables in Book of Concord citation text chunks and render them as beautifully styled HTML tables in the UI.

**Architecture:** Add a utility helper function in `pipeline/orchestrator.py` to parse markdown tables into custom-styled HTML table tags while keeping standard paragraphs formatted as `<p>` blocks. Inject styling into `ui/app.py`.

**Tech Stack:** Python, Streamlit, HTML/CSS.

---

### Task 1: Add Markdown Table Parser Helper and Update format_deep_dive_details

**Files:**
- Modify: `pipeline/orchestrator.py`
- Test: `tests/pipeline/test_orchestrator_execution.py`

**Step 1: Write the failing tests**
Write `test_format_boc_text_with_table` and `test_format_boc_text_mixed` in `tests/pipeline/test_orchestrator_execution.py`.

**Step 2: Run tests to verify they fail**
Run: `pytest tests/pipeline/test_orchestrator_execution.py`

**Step 3: Write minimal implementation**
Implement `format_boc_text` and update `format_deep_dive_details` in `pipeline/orchestrator.py`.

**Step 4: Run tests to verify they pass**
Run: `pytest tests/pipeline/test_orchestrator_execution.py`

**Step 5: Commit**
`git add pipeline/orchestrator.py tests/pipeline/test_orchestrator_execution.py`
`git commit -m "feat: add markdown table parser to orchestrator and verify with unit tests"`

---

### Task 2: Inject Custom CSS Styles for Tables

**Files:**
- Modify: `ui/app.py`

**Step 1: Write implementation**
Add `.boc-table` CSS rules into `st.markdown` inside `ui/app.py`.

**Step 2: Manual Verification**
Run: `streamlit run ui/app.py` (Verify visually using manual steps)

**Step 3: Commit**
`git add ui/app.py`
`git commit -m "feat: style Book of Concord HTML tables in Streamlit app"`
