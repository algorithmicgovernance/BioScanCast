"""LLM-driven query decomposition and question-type classification.

Given a ForecastQuestion, this module:
  1. Classifies the question type (outbreak_count, binary_event, etc.)
  2. Decomposes it into 5-8 search-engine-optimised sub-queries
  3. Validates sub-query word counts (2-8 words) post-hoc
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import List

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.llm.client import LLMClient

logger = logging.getLogger(__name__)

VALID_AXES: set[str] = {
    "latest_data",
    "trend",
    "mechanism",
    "policy",
    "historical_analogy",
    "expert_opinion",
}

QUESTION_TYPES: set[str] = {
    "outbreak_count",
    "binary_event",
    "mechanism_or_attribution",
    "unknown",
}

# Which axes each question type should use
AXES_BY_TYPE: dict[str, list[str]] = {
    "outbreak_count": ["latest_data", "trend", "policy", "historical_analogy"],
    "binary_event": ["latest_data", "trend", "mechanism", "expert_opinion", "historical_analogy"],
    "mechanism_or_attribution": ["mechanism", "expert_opinion", "latest_data"],
    "unknown": list(VALID_AXES),
}


@dataclass
class SubQuery:
    id: str
    question_id: str
    text: str
    axis: str


def classify_question_type(question: ForecastQuestion, llm_client: LLMClient) -> str:
    """Classify a forecast question into one of the known question types.

    Design decision: uses an LLM call rather than keyword heuristics.
    The LLM approach is more flexible for novel question phrasings, and the
    cost is negligible (one small JSON call).  Falls back to "unknown" on
    any failure, which routes to all axes — safe but slightly wasteful.
    Revisit if classification latency or cost becomes an issue.
    """
    prompt = json.dumps(
        {
            "task": (
                "Classify this biosecurity forecast question into exactly one type. "
                "Return JSON: {\"question_type\": \"<type>\"}. "
                "Types: outbreak_count (how many cases by date X), "
                "binary_event (will event X occur by date Y), "
                "mechanism_or_attribution (what caused X), "
                "unknown (if none fit clearly)."
            ),
            "question": question.text,
            "pathogen": question.pathogen,
            "event_type": question.event_type,
        }
    )
    try:
        result = llm_client.generate_json(prompt)
        qtype = result.get("question_type", "unknown")
        if qtype not in QUESTION_TYPES:
            logger.warning("LLM returned unknown question type '%s', falling back to 'unknown'", qtype)
            return "unknown"
        return qtype
    except Exception:
        logger.exception("Question classification failed, defaulting to 'unknown'")
        return "unknown"


def _build_decomposition_prompt(question: ForecastQuestion, question_type: str) -> str:
    axes = AXES_BY_TYPE.get(question_type, list(VALID_AXES))
    return json.dumps(
        {
            "task": (
                "Decompose this biosecurity forecast question into 5-8 search-engine-optimised "
                "sub-queries. Each sub-query should be 2-8 words and target a specific information "
                "axis. Return strict JSON: {\"sub_queries\": [{\"text\": \"...\", \"axis\": \"...\"}]}. "
                "No prose."
            ),
            "question": question.text,
            "pathogen": question.pathogen,
            "region": question.region,
            "target_date": question.target_date.isoformat() if question.target_date else None,
            "allowed_axes": axes,
        }
    )


def _validate_word_count(text: str) -> str | None:
    """Return trimmed text if 2-8 words, else None."""
    words = text.strip().split()
    if 2 <= len(words) <= 8:
        return " ".join(words)
    if len(words) > 8:
        truncated = " ".join(words[:8])
        logger.warning("Sub-query truncated from %d to 8 words: '%s'", len(words), truncated)
        return truncated
    logger.warning("Sub-query dropped (too short, %d words): '%s'", len(words), text)
    return None


def _fallback_subqueries(question: ForecastQuestion) -> List[SubQuery]:
    """Generate simple keyword-based sub-queries when LLM decomposition fails."""
    queries: list[SubQuery] = []
    base_terms = []
    if question.pathogen:
        base_terms.append(question.pathogen)
    if question.region:
        base_terms.append(question.region)

    fallbacks = [
        (" ".join(base_terms + ["latest cases 2024 2025"]), "latest_data"),
        (" ".join(base_terms + ["outbreak trend"]), "trend"),
        (" ".join(base_terms + ["government response policy"]), "policy"),
    ]
    if not base_terms:
        # Use question text fragments
        words = question.text.split()[:4]
        fallbacks = [
            (" ".join(words + ["latest"]), "latest_data"),
            (" ".join(words + ["trend"]), "trend"),
            (" ".join(words + ["expert analysis"]), "expert_opinion"),
        ]

    for text, axis in fallbacks:
        validated = _validate_word_count(text)
        if validated:
            queries.append(
                SubQuery(
                    id=uuid.uuid4().hex,
                    question_id=question.id,
                    text=validated,
                    axis=axis,
                )
            )
    return queries


def decompose_question(
    question: ForecastQuestion, llm_client: LLMClient
) -> List[SubQuery]:
    """Decompose a forecast question into sub-queries using an LLM.

    Falls back to simple keyword-based sub-queries if the LLM fails.
    """
    question_type = classify_question_type(question, llm_client)
    prompt = _build_decomposition_prompt(question, question_type)

    try:
        result = llm_client.generate_json(prompt)
    except Exception:
        logger.exception("LLM decomposition failed, using fallback sub-queries")
        return _fallback_subqueries(question)

    raw_queries = result.get("sub_queries", [])
    if not isinstance(raw_queries, list):
        logger.warning("LLM returned non-list sub_queries, using fallback")
        return _fallback_subqueries(question)

    sub_queries: list[SubQuery] = []
    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        axis = item.get("axis", "")
        if axis not in VALID_AXES:
            logger.warning("Dropping sub-query with invalid axis '%s': '%s'", axis, text)
            continue
        validated = _validate_word_count(text)
        if validated is None:
            continue
        sub_queries.append(
            SubQuery(
                id=uuid.uuid4().hex,
                question_id=question.id,
                text=validated,
                axis=axis,
            )
        )

    if len(sub_queries) < 3:
        logger.warning(
            "Only %d valid sub-queries from LLM (need >=3), supplementing with fallback",
            len(sub_queries),
        )
        sub_queries.extend(_fallback_subqueries(question))

    # Cap at 8
    return sub_queries[:8]
