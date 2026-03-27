from __future__ import annotations

import json
from typing import Dict, List, Protocol

from .models import FilterDecision, ForecastQuestion, SearchResult


class LLMClient(Protocol):
    def generate_json(self, prompt: str) -> dict:
        ...


def build_filter_prompt(
    question: ForecastQuestion,
    candidates: list[dict],
) -> str:
    payload = {
        "task": (
            "You are filtering search results for a biosecurity forecasting pipeline. "
            "Keep only candidates likely to contain relevant factual evidence for forecasting. "
            "Prefer official, primary, recent, and event-specific sources. "
            "Reject low-information, generic, duplicated, or weakly relevant pages."
        ),
        "question": {
            "id": question.id,
            "text": question.text,
            "region": question.region,
            "pathogen": question.pathogen,
            "event_type": question.event_type,
            "resolution_criteria": question.resolution_criteria,
        },
        "candidates": candidates,
        "output_schema": {
            "decisions": [
                {
                    "result_id": "string",
                    "keep": "boolean",
                    "relevance_score": "0_to_1_float",
                    "credibility_score": "0_to_1_float",
                    "final_score": "0_to_1_float",
                    "reason_codes": ["list_of_short_strings"],
                    "notes": "short explanation",
                }
            ]
        },
    }
    return json.dumps(payload, default=str, indent=2)


def llm_filter_candidates(
    question: ForecastQuestion,
    candidate_decisions: List[FilterDecision],
    result_map: Dict[str, SearchResult],
    llm_client: LLMClient,
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

    prompt = build_filter_prompt(question, candidates)
    response = llm_client.generate_json(prompt)

    output_by_id = {item["result_id"]: item for item in response.get("decisions", [])}

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