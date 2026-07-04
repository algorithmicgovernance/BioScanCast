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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bioscancast.schemas import Document, InsightRecord
from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.llm.base import LLMClient

from .budget import BudgetTracker
from .config import InsightConfig
from .retrieval.hybrid import hybrid_retrieve
from .text_extraction.chunk_extractor import extract_facts_from_chunk

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

    def _safe_extract(self, sc, doc, question, config):
        """Wrap chunk extraction so a failure in one chunk doesn't kill
        the document. Returns the (records, response) tuple or None.

        Called both serially and from ThreadPoolExecutor workers — must
        not mutate ``self`` so concurrent calls stay safe.
        """
        try:
            return extract_facts_from_chunk(
                sc.chunk,
                doc,
                question,
                self._llm,
                model=config.cheap_model,
                max_tokens=config.extraction_max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Chunk extraction failed (chunk_id=%s): %s",
                sc.chunk.chunk_id, exc,
            )
            return None

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

        # Adaptive top-k: when the filter passes through only a handful
        # of usable documents, lift retrieval depth so the per-doc chunk
        # budget isn't the bottleneck on coverage. See InsightConfig
        # docstrings for the rationale.
        usable_doc_count = sum(
            1 for d in documents if d.status != "failed" and d.chunks
        )
        if usable_doc_count <= config.low_survival_doc_threshold:
            effective_top_k = max(config.retrieval_top_k, config.low_survival_top_k)
            effective_max_chunks = max(
                config.max_chunks_per_document, config.low_survival_top_k
            )
            if effective_top_k != config.retrieval_top_k:
                result.notes.append(
                    f"Low-survival adaptive top_k engaged: "
                    f"{usable_doc_count} usable docs (≤ threshold "
                    f"{config.low_survival_doc_threshold}); "
                    f"retrieval_top_k={effective_top_k} (default "
                    f"{config.retrieval_top_k})."
                )
        else:
            effective_top_k = config.retrieval_top_k
            effective_max_chunks = config.max_chunks_per_document

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
                top_k=effective_top_k,
                bm25_weight=config.bm25_weight,
                embedding_weight=config.embedding_weight,
                embedding_model=config.embedding_model,
                embedding_cache=embedding_cache,
            )

            # Cap chunks per document
            scored_chunks = scored_chunks[:effective_max_chunks]

            # --- Per-chunk extraction (parallel within a doc) ---
            # Live tests on real biosecurity documents show the per-doc
            # wall-clock is almost entirely sequential OpenAI request
            # latency. A ThreadPoolExecutor cuts WHO mpox (~30s) and ECDC
            # CDTR (~38s) down by roughly chunk_workers× because each
            # request is independent and the OpenAI sync client is
            # thread-safe. Errors in one chunk don't kill the doc;
            # budget accounting happens serially after futures complete
            # so BudgetTracker doesn't need its own lock.
            workers = max(1, min(config.chunk_workers, len(scored_chunks)))
            if workers == 1 or len(scored_chunks) <= 1:
                chunk_results = [
                    self._safe_extract(sc, doc, question, config)
                    for sc in scored_chunks
                ]
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [
                        ex.submit(
                            self._safe_extract, sc, doc, question, config,
                        )
                        for sc in scored_chunks
                    ]
                    chunk_results = [f.result() for f in futures]

            for outcome in chunk_results:
                if outcome is None:
                    continue
                records, response = outcome
                budget.record(response)
                all_records.extend(records)

            result.documents_processed += 1

        # --- Cross-document deduplication ---
        all_records = _deduplicate_records(all_records)

        result.records = all_records
        result.budget_summary = budget.summary()
        return result


def _normalize_location(location: Optional[str]) -> str:
    """Normalize location for dedup comparison."""
    if not location:
        return ""
    return location.lower().strip()


# Ordered from coarsest to finest. Used by the dedup logic to compare
# two records whose event_date_precision values differ.
_PRECISION_ORDER = {"year": 0, "month": 1, "day": 2}


def _date_bucket(
    dt: Optional[datetime], precision: Optional[str]
) -> Optional[tuple]:
    """Truncate a (datetime, precision) pair to a comparable tuple bucket.

    Returns ``None`` when no date is known.
    """
    if dt is None or precision is None:
        return None
    if precision == "year":
        return (dt.year,)
    if precision == "month":
        return (dt.year, dt.month)
    # day (or anything more specific) is collapsed to day
    return (dt.year, dt.month, dt.day)


def _dates_overlap(
    d1: Optional[datetime],
    p1: Optional[str],
    d2: Optional[datetime],
    p2: Optional[str],
) -> bool:
    """Check whether two (datetime, precision) pairs refer to
    overlapping time buckets.

    Both-None counts as overlap (the "no date known" bucket). One-None
    does NOT overlap with a known-date bucket — we don't merge dated
    and undated facts.

    Two known dates overlap when, truncated to whichever precision is
    coarser, their buckets are equal. So month-precision ``2026-01``
    overlaps with day-precision ``2026-01-25`` (both → ``(2026, 1)``
    at month precision) but not with ``2026-02-25``.
    """
    if d1 is None and d2 is None:
        return True
    if d1 is None or d2 is None:
        return False
    if p1 is None or p2 is None:
        return False
    coarser = p1 if _PRECISION_ORDER[p1] <= _PRECISION_ORDER[p2] else p2
    return _date_bucket(d1, coarser) == _date_bucket(d2, coarser)


def _values_compatible(
    v1: Optional[float], v2: Optional[float], *, rel_tol: float = 0.01
) -> bool:
    """Check whether two metric_values are close enough to be the same fact.

    Both-None counts as compatible (the "no value" bucket). One-None is
    compatible with a known value (one source happened to omit the
    count). Two known values are compatible when their relative
    difference is within ``rel_tol`` (default 1%) — this accommodates
    rounding (e.g. "6500" vs "6543") without merging genuinely
    different counts (e.g. African Region 9782 vs DRC 6543 — which
    happens when the model misattributes a regional quote to a
    specific country).
    """
    if v1 is None or v2 is None:
        return True
    if v1 == v2:
        return True
    # Relative difference; guard against division by zero
    denom = max(abs(v1), abs(v2))
    if denom == 0:
        return v1 == v2
    return abs(v1 - v2) / denom <= rel_tol


def _record_dedup_key(record: InsightRecord) -> tuple:
    """First-stage dedup key: groups records that *might* be duplicates.

    Date is intentionally omitted from the first-stage key because two
    records with different date precisions (e.g. ``2026-01`` vs
    ``2026-01-25``) need to be considered together. The second stage
    walks each group and uses ``_dates_overlap`` to decide whether to
    merge.
    """
    return (
        record.event_type,
        record.metric_name or "",
        _normalize_location(record.location),
    )


def _merge_record_into(
    target: InsightRecord, source: InsightRecord
) -> None:
    """Merge ``source`` into ``target`` in place.

    Adds source's unique chunk references, raises confidence to the max
    of the two, and adopts the finer of the two date precisions (with
    its corresponding date). The coarser-precision source loses its
    date but its provenance is preserved.
    """
    existing_chunk_ids = {
        (s.document_id, s.chunk_id) for s in target.sources
    }
    for src in source.sources:
        if (src.document_id, src.chunk_id) not in existing_chunk_ids:
            target.sources.append(src)
    if source.confidence > target.confidence:
        target.confidence = source.confidence
    # Adopt the finer precision date if source has one
    source_rank = (
        _PRECISION_ORDER.get(source.event_date_precision, -1)
        if source.event_date_precision else -1
    )
    target_rank = (
        _PRECISION_ORDER.get(target.event_date_precision, -1)
        if target.event_date_precision else -1
    )
    if source.event_date and source_rank > target_rank:
        target.event_date = source.event_date
        target.event_date_precision = source.event_date_precision


def _deduplicate_records(records: list[InsightRecord]) -> list[InsightRecord]:
    """Two-stage deduplication merging records with overlapping date buckets.

    Stage 1: group by ``(event_type, metric_name, normalized_location)``.
    Stage 2: within each group, walk records in order and merge each
    into the first surviving entry whose date bucket overlaps. This
    handles the common case where multiple sources report the same
    event at different date precisions (e.g. WHO sitrep says
    "January 2026" while a country report says "as of 25 January
    2026") — both refer to the same underlying fact.

    Records with completely distinct dates within the same group stay
    separate (no false merging of "Jan 5" with "Jan 6").
    """
    from collections import defaultdict

    groups: dict[tuple, list[InsightRecord]] = defaultdict(list)
    for record in records:
        groups[_record_dedup_key(record)].append(record)

    out: list[InsightRecord] = []
    for group in groups.values():
        merged: list[InsightRecord] = []
        for record in group:
            target = None
            for existing in merged:
                if not _dates_overlap(
                    record.event_date, record.event_date_precision,
                    existing.event_date, existing.event_date_precision,
                ):
                    continue
                if not _values_compatible(
                    record.metric_value, existing.metric_value,
                ):
                    # Same dedup key + overlapping dates but different
                    # numeric values — almost always a model attribution
                    # error (e.g. regional total mistakenly tagged with
                    # a country location). Keep both records so the
                    # conflict is visible downstream rather than
                    # silently dropped.
                    continue
                target = existing
                break
            if target is not None:
                _merge_record_into(target, record)
            else:
                merged.append(record)
        out.extend(merged)
    return out
