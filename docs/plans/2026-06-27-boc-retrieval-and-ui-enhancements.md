# RAG Confessional Hits, Collapsible UI, and Hermeneutical Lens Implementation Plan

> **For Gemini:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modify configuration to retrieve top 5 Book of Concord passages, implement collapsible HTML nested details in the UI block, and instruct the LLM system prompt to interpret scriptures confessionaly via the hermeneutical circle.

**Architecture:** 
1. Update default `rag_confessional_k` in `config/settings.py` to `5`.
2. Enhance `pipeline/orchestrator.py` `format_deep_dive_details` to return nested `<details>` elements for each confessional chunk and update tests.
3. Inject hermeneutical instructions in `pipeline/prompt.py` guiding the model to interpret scriptures using the Book of Concord.
4. Add CSS styles to `ui/app.py` for nested `details` tags.

**Tech Stack:** Python, Streamlit, HTML/CSS, Pytest

---

### Task 1: Update Configuration settings

**Files:**
- Modify: [settings.py](file:///c:/dev/IdeaProjects/lutheran-llm/config/settings.py)

**Step 1: Write the failing test**
We will add a test verifying settings default values.
Modify: `tests/config/test_settings.py` (or create if not exist) to assert `rag_confessional_k == 5`.

**Step 2: Run test to verify it fails**
Run: `pytest tests/config/test_settings.py`
Expected: FAIL if it was 2.

**Step 3: Write minimal implementation**
Modify [settings.py](file:///c:/dev/IdeaProjects/lutheran-llm/config/settings.py):
```python
    rag_confessional_k: int = 5
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/config/test_settings.py`
Expected: PASS

**Step 5: Commit**
```bash
git add config/settings.py tests/config/test_settings.py
git commit -m "config: set default Book of Concord retrieval k to 5"
```

---

### Task 2: Update System Prompt for Hermeneutical Circle

**Files:**
- Modify: [prompt.py](file:///c:/dev/IdeaProjects/lutheran-llm/pipeline/prompt.py)

**Step 1: Write the failing test**
Modify `tests/pipeline/test_prompt.py` (or check if exists) to verify that `SYSTEM_PROMPT` contains keywords like "lens" or "hermeneutical".

**Step 2: Run test to verify it fails**
Run: `pytest tests/pipeline/test_prompt.py`
Expected: FAIL

**Step 3: Write minimal implementation**
Modify [prompt.py](file:///c:/dev/IdeaProjects/lutheran-llm/pipeline/prompt.py) to add instruction 5:
```python
SYSTEM_PROMPT = """You are a strictly orthodox confessional Lutheran AI assistant.
Your objective is to provide clear, faithful, and scripturally grounded answers to inquiries about the Lutheran faith.

CRITICAL INSTRUCTIONS:
1. Base your assertions exclusively on the verified text snippets provided to you from Holy Scripture and the Book of Concord. Do not invent, extrapolate, or introduce heterodox teachings.
2. If the provided context is silent on a speculative matter, explicitly state that Scripture does not reveal an answer.
3. If a query indicates intense personal guilt, spiritual crisis, or a need for pastoral counseling, provide immediate comforting Gospel assurance and direct the user to consult a local pastor.
4. Always use the Lutheran numbering of the Ten Commandments as taught in Luther's Small Catechism...
5. Recognize that Holy Scripture is the source and norm (norma normans) for the Book of Concord, but the Book of Concord is the correct confession and hermeneutical lens (norma normata) through which Scripture must be interpreted. Apply this hermeneutical circle: always interpret and synthesize biblical passages through the confessional lens of the Book of Concord, specifically applying the Law/Gospel distinction to ensure retrieved Scripture is not presented out of theological context.
...
"""
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/pipeline/test_prompt.py`
Expected: PASS

**Step 5: Commit**
```bash
git add pipeline/prompt.py tests/pipeline/test_prompt.py
git commit -m "prompt: add hermeneutical circle guidance for scripture interpretation"
```

---

### Task 3: Update Collapsible UI in Orchestrator and Streamlit Styling

**Files:**
- Modify: [orchestrator.py](file:///c:/dev/IdeaProjects/lutheran-llm/pipeline/orchestrator.py)
- Modify: [app.py](file:///c:/dev/IdeaProjects/lutheran-llm/ui/app.py)

**Step 1: Write the failing test**
Modify `tests/pipeline/test_orchestrator_execution.py` to verify that `format_deep_dive_details` outputs details elements with citation and class for book of concord.

**Step 2: Run test to verify it fails**
Run: `pytest tests/pipeline/test_orchestrator_execution.py`
Expected: FAIL

**Step 3: Write minimal implementation**
Modify [orchestrator.py](file:///c:/dev/IdeaProjects/lutheran-llm/pipeline/orchestrator.py) in `format_deep_dive_details`:
```python
    # Book of Concord Citations
    lines.append("<h4>Book of Concord Citations</h4>")
    confessional = retrieved_ctx.get("confessional", [])
    if confessional:
        for chunk in confessional:
            citation = html.escape(chunk.get("citation", "Book of Concord"))
            text_val = html.escape(chunk.get("text", ""))
            lines.append(
                f'<details class="boc-detail" style="margin-bottom: 0.6rem; margin-left: 0.5rem; border-left: 2px solid #F59E0B; padding: 0.3rem 0.6rem;">'
                f'  <summary style="font-weight: 500; font-size: 0.95rem; color: #E2E8F0; cursor: pointer;">{citation}</summary>'
                f'  <p style="margin-top: 0.4rem; font-style: italic; color: #94A3B8; font-size: 0.9rem;">"{text_val}"</p>'
                f'</details>'
            )
```

Also, modify [app.py](file:///c:/dev/IdeaProjects/lutheran-llm/ui/app.py) to add custom details styling if necessary in CSS.

**Step 4: Run test to verify it passes**
Run: `pytest tests/pipeline/test_orchestrator_execution.py`
Expected: PASS

**Step 5: Commit**
```bash
git add pipeline/orchestrator.py ui/app.py tests/pipeline/test_orchestrator_execution.py
git commit -m "ui: make Book of Concord citations collapsible using nested details"
```

---

## Verification Plan

### Automated Tests
- Run `pytest` on all updated/created tests:
  `pytest tests/`

### Manual Verification
- Launch the Streamlit application using `streamlit run ui/app.py`.
- Run queries (e.g., "how are we justified?") and confirm that:
  1. 5 Book of Concord citations are shown in the Theological Depth section.
  2. Each Book of Concord passage is collapsed by default and reveals the full text when the citation is clicked.
  3. The response synthesizes the scriptural context through the Lutheran lens of the Book of Concord.
