from __future__ import annotations

import json
from typing import Dict, List

from bioscancast.llm.base import LLMClient

from .models import FilterDecision, ForecastQuestion, SearchResult


# Default model and max output tokens for the filter LLM call. These can be
# overridden via `llm_filter_candidates(... , model=..., max_tokens=...)`.
DEFAULT_FILTER_MODEL = "gpt-4o-mini"
DEFAULT_FILTER_MAX_TOKENS = 4096


FILTER_SYSTEM_PROMPT = (
    "You are filtering search results for a biosecurity forecasting "
    "pipeline. Your job is to decide which candidates are likely to "
    "contain relevant factual evidence for forecasting. Prefer official, "
    "primary, recent, and event-specific sources. Reject low-information, "
    "generic, duplicated, or weakly relevant pages. Return a JSON object "
    "matching the supplied schema with one decision per candidate."
)


FILTER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "result_id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "relevance_score": {
                        "type": "number", "minimum": 0.0, "maximum": 1.0,
                    },
                    "credibility_score": {
                        "type": "number", "minimum": 0.0, "maximum": 1.0,
                    },
                    "final_score": {
                        "type": "number", "minimum": 0.0, "maximum": 1.0,
                    },
                    "reason_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "notes": {"type": ["string", "null"]},
                },
                "required": [
                    "result_id", "keep", "relevance_score",
                    "credibility_score", "final_score", "reason_codes",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def build_filter_prompt(
    question: ForecastQuestion,
    candidates: list[dict],
) -> tuple[str, str, dict]:
    """Build the (system, user, schema) tuple for the filter LLM call.

    The user prompt is a JSON payload containing the question and the
    candidate list; the system prompt covers the task instructions; the
    schema is enforced by structured-output capable LLMs (and ignored
    by ones that aren't).
    """
    user_payload = {
        "question": {
            "id": question.id,
            "text": question.text,
            "region": question.region,
            "pathogen": question.pathogen,
            "event_type": question.event_type,
            "resolution_criteria": question.resolution_criteria,
        },
        "candidates": candidates,
    }
    user = json.dumps(user_payload, default=str, indent=2)
    return FILTER_SYSTEM_PROMPT, user, FILTER_OUTPUT_SCHEMA


def llm_filter_candidates(
    question: ForecastQuestion,
    candidate_decisions: List[FilterDecision],
    result_map: Dict[str, SearchResult],
    llm_client: LLMClient,
    *,
    model: str = DEFAULT_FILTER_MODEL,
    max_tokens: int = DEFAULT_FILTER_MAX_TOKENS,
) -> List[FilterDecision]:
    if not candidate_decisions:
        return []

    candidates = []
    for decision in candidate_decisions:
        result = result_map[decision.result_id]
        candidates.append(
            {
                "result_id": result.id,
                "url": result.url,
                "domain": result.domain,
                "title": result.title,
                "snippet": result.snippet,
                "published_date": result.published_date.isoformat() if result.published_date else None,
                "file_type": result.file_type,
                "source_tier": result.source_tier,
                "is_official_domain": result.is_official_domain,
                "search_stage_score": result.search_stage_score,
            }
        )

    system, user, schema = build_filter_prompt(question, candidates)
    response = llm_client.generate_json(
        system=system,
        user=user,
        schema=schema,
        model=model,
        max_tokens=max_tokens,
    )

    output_by_id = {
        item["result_id"]: item
        for item in response.content.get("decisions", [])
    }

    updated: list[FilterDecision] = []
    for decision in candidate_decisions:
        data = output_by_id.get(decision.result_id)
        if not data:
            decision.keep = False
            decision.stage = "llm"
            decision.reason_codes.append("missing_llm_decision")
            updated.append(decision)
            continue

        decision.keep = bool(data.get("keep", False))
        decision.stage = "llm"
        decision.relevance_score = float(data.get("relevance_score", decision.relevance_score))
        decision.credibility_score = float(data.get("credibility_score", decision.credibility_score))
        decision.priority_score = float(data.get("final_score", decision.priority_score))
        decision.reason_codes = list(data.get("reason_codes", decision.reason_codes))
        decision.notes = data.get("notes")
        updated.append(decision)

    return updated
