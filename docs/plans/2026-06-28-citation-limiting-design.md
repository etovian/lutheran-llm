# Design: Scripture and Confessional Citation Limiting

## Goal
Limit the Scripture passages and Book of Concord citations displayed in the collapsible `<details>` ("Theological Depth") block at the end of responses to only those actually cited in the LLM's response body. This ensures the reference section remains relevant and does not include unreferenced search results.

## Approach: Reference Labeling & Response Post-Processing

### 1. Labeling Context in the System Prompt
When assembling the context to pass to the LLM, we will assign a unique identifier `[Ref-N]` (e.g., `[Ref-1]`, `[Ref-2]`, etc.) to each retrieved item:
* We number all items sequentially across both confessional chunks and scripture passages.
* Each item in the system prompt context will be clearly prefixed with its `[Ref-N]` label.

### 2. Instructing the LLM
We will update `SYSTEM_PROMPT` to require the LLM to place a special block at the very end of its response containing only the reference labels it actually utilized.
* Format: `<citations>[Ref-1], [Ref-2]</citations>`

### 3. Orchestrator Post-Processing
In `pipeline/orchestrator.py`:
* Parse the LLM's response to extract any `[Ref-N]` codes inside the `<citations>...</citations>` block.
* Validate these codes against the list of retrieved items (discarding any codes that were not part of the input context to prevent hallucinations).
* Strip the `<citations>...</citations>` block entirely from the main response body so it remains completely hidden from the user.
* Filter the `retrieved_ctx` dictionary to keep only the cited items.
* Pass the filtered context to `format_deep_dive_details` so the collapsible block displays only the cited passages.

### 4. Robust Fallback
* If no citations block is found, or if parsing fails completely, the orchestrator will fall back to using the full list of retrieved scriptures and confessional chunks, ensuring we do not fail to show the details section.
