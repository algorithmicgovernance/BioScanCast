# Insight Stage

The insight stage turns a forecast question plus a list of `Document` objects into a list of `InsightRecord` objects — the structured "dataframe of facts" the grant proposal refers to.

## Flow

```
Documents + ForecastQuestion
    |
    v
[1] Skip failed/empty documents
    |
    v
[2] Hybrid retrieval (BM25 + embeddings + rule-based soft re-rank)
    -> top-k chunks per document
    |
    v
[3] Per-chunk LLM extraction (cheap model, one chunk at a time)
    -> zero or more InsightRecords per chunk
    -> hallucination guard: quote must be substring of chunk text
    |
    v
[4] Budget check after each document (stop early if exceeded)
    |
    v
[5] Cross-document deduplication (merge provenance, keep higher confidence)
    |
    v
InsightRunResult (records + budget_summary + notes)
```

## Key design decisions

- **BM25 + embeddings**: Pure embedding similarity is bad at finding numbers. BM25 is the antidote. Both are used in hybrid retrieval.
- **Per-chunk extraction**: One chunk per LLM call. Simpler prompts, easier hallucination control. Don't batch chunks without a measured reason.
- **Hallucination guard**: Verbatim quote substring check. Cheap and surprisingly effective at catching training-data contamination. Don't soften to fuzzy match.
- **In-memory numpy for embeddings**: No vector database needed at this scale (dozens to low-hundreds of chunks per question).
- **Soft rule filters**: Date/keyword filters bias retrieval but don't hard-drop chunks by default.

## Configuration

See `config.py` for all configurable values (`InsightConfig`).

## TODO

- [ ] Migrate `bioscancast/filtering/llm_filter.py`'s local `LLMClient` protocol to the shared `bioscancast/llm/base.py` protocol.
- [ ] Cognitive bias mitigations belong primarily in the **forecasting** stage's prompts, not here. Insight extraction is neutral fact-finding. A brief reminder is included in the extraction prompt but full bias mitigation should be implemented in forecasting.
- [ ] When extraction lands and synthetic fixtures are swapped for real Document outputs, expect surprises. Plan a follow-up PR to harden the chunk extractor against messy real-world chunks (long text, mid-sentence breaks, OCR garbage in tables).
- [ ] Strong model refinement pass (behind `use_strong_model_refinement` config flag, currently a no-op).
