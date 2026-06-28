# Scripture Relevancy Threshold Design

This document details the design for filtering retrieved Scripture passages using a configurable relevancy threshold rather than returning a fixed limit of 10.

## User Review Required

> [!NOTE]
> We decided to start with **Approach A**: filtering using Chroma's default L2 distance space. The distance threshold will be configurable so we can tune it to get the best results.

> [!IMPORTANT]
> If no scriptures meet the threshold, the system will return **strict 0 passages** (no fallback).

## Proposed Changes

### Configuration

We will add the following configuration settings to `config/settings.py` (with defaults and environment variable support):

* `rag_biblical_max_pool`: Maximum number of passages to fetch from Chroma DB (default: `50`).
* `rag_biblical_distance_threshold`: Max allowed L2 distance for a passage to be included (default: `1.2`).

### Context Retrieval

In `pipeline/orchestrator.py#retrieve_context`:
* Instead of querying with `n_results=biblical_k`, we will query with `n_results=settings.rag_biblical_max_pool`.
* We will extract the `"distances"` from Chroma's query response.
* We will filter out any biblical passage whose distance is strictly greater than `settings.rag_biblical_distance_threshold`.
* We will log the filtering details for tuning purposes.

### Verification Plan

#### Automated Tests
* Update `tests/pipeline/test_orchestrator_retrieval.py` to assert that:
  * When threshold is high (e.g. `2.0`), all retrieved passages are returned.
  * When threshold is low (e.g. `0.1`), zero passages are returned.
