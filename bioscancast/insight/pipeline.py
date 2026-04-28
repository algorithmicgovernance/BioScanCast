"""Insight pipeline orchestrator.

Turns a forecast question plus a list of Document objects into a list
of InsightRecord objects — the structured "dataframe of facts" the
grant proposal refers to.

Per-document flow:
1. Skip documents with status="failed" or empty chunks.
2. hybrid_retrieve to get top-k chunks for this document.
3. Per retrieved chunk: extract_facts_from_chunk with the cheap model.
4. Collect InsightRecords, record budget.
5. Stop early if budget exceeded.

After all documents:
6. Deduplicate facts across documents.
7. (Optional) Strong model refinement (default off).
8. Return InsightRunResult.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bioscancast.schemas import Document, InsightRecord
from bioscancast.filtering.models import ForecastQuestion
from bioscancast.llm.base import LLMClient

from .budget import BudgetTracker
from .config import InsightConfig
from .retrieval.hybrid import hybrid_retrieve
from .extraction.chunk_extractor import extract_facts_from_chunk

logger = logging.getLogger(__name__)


@dataclass
class InsightRunResult:
    """Result of a full insight pipeline run."""

    records: list[InsightRecord] = field(default_factory=list)
    budget_summary: dict = field(default_factory=dict)
    documents_processed: int = 0
    documents_skipped: int = 0
    notes: list[str] = field(default_factory=list)


class InsightPipeline:
    """Orchestrates chunk retrieval and LLM extraction for the insight stage."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        config: Optional[InsightConfig] = None,
    ) -> None:
        self._llm = llm_client
        self._config = config or InsightConfig()

    def run(
        self,
        question: ForecastQuestion,
        documents: list[Document],
    ) -> InsightRunResult:
        """Run the insight pipeline over a list of documents.

        Args:
            question: The forecast question driving extraction.
            documents: Documents to process (from extraction stage).

        Returns:
            InsightRunResult with extracted records and budget info.
        """
        config = self._config
        budget = BudgetTracker()
        all_records: list[InsightRecord] = []
        result = InsightRunResult()
        embedding_cache: dict[str, list[float]] = {}

        for doc in documents:
            # --- Skip check ---
            if doc.status == "failed" or not doc.chunks:
                result.documents_skipped += 1
                reason = "failed" if doc.status == "failed" else "no chunks"
                result.notes.append(
                    f"Skipped document {doc.id}: {reason}"
                )
                continue

            # --- Budget check (before processing) ---
            if budget.would_exceed(config.max_input_tokens_per_run):
                result.notes.append(
                    f"Budget exceeded ({budget.total_input_tokens} input tokens). "
                    f"Stopping early at document {doc.id}."
                )
                break

            # --- Retrieval ---
            scored_chunks = hybrid_retrieve(
                question,
                doc,
                self._llm,
                top_k=config.retrieval_top_k,
                bm25_weight=config.bm25_weight,
                embedding_weight=config.embedding_weight,
                embedding_model=config.embedding_model,
                embedding_cache=embedding_cache,
            )

            # Cap chunks per document
            scored_chunks = scored_chunks[: config.max_chunks_per_document]

            # --- Per-chunk extraction ---
            for sc in scored_chunks:
                records, response = extract_facts_from_chunk(
                    sc.chunk,
                    doc,
                    question,
                    self._llm,
                    model=config.cheap_model,
                )
                budget.record(response)
                all_records.extend(records)

            result.documents_processed += 1

        # --- Cross-document deduplication ---
        all_records = _deduplicate_records(all_records)

        # --- Optional strong model refinement ---
        if config.use_strong_model_refinement:
            result.notes.append(
                "Strong model refinement is enabled but not yet implemented."
            )

        result.records = all_records
        result.budget_summary = budget.summary()
        return result


def _normalize_location(location: Optional[str]) -> str:
    """Normalize location for dedup comparison."""
    if not location:
        return ""
    return location.lower().strip()


def _record_dedup_key(record: InsightRecord) -> tuple:
    """Build a deduplication key for an InsightRecord.

    Two records are duplicates if they have the same event_type,
    metric_name, date, and normalized location.
    """
    date_str = ""
    if record.event_date:
        date_str = record.event_date.strftime("%Y-%m-%d")
    return (
        record.event_type,
        record.metric_name or "",
        date_str,
        _normalize_location(record.location),
    )


def _deduplicate_records(records: list[InsightRecord]) -> list[InsightRecord]:
    """Deduplicate InsightRecords, merging provenance lists.

    Keeps the record with the higher confidence score and merges
    source references from duplicates.
    """
    seen: dict[tuple, InsightRecord] = {}

    for record in records:
        key = _record_dedup_key(record)
        if key in seen:
            existing = seen[key]
            # Merge provenance
            existing_chunk_ids = {
                (s.document_id, s.chunk_id) for s in existing.sources
            }
            for src in record.sources:
                if (src.document_id, src.chunk_id) not in existing_chunk_ids:
                    existing.sources.append(src)
            # Keep higher confidence
            if record.confidence > existing.confidence:
                existing.confidence = record.confidence
        else:
            seen[key] = record

    return list(seen.values())
