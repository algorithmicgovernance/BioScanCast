"""Tests for the LLM-driven filtering stage.

The filter uses the shared ``bioscancast.llm.base.LLMClient`` protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.filtering.llm_filter import (
    DEFAULT_FILTER_MAX_TOKENS,
    DEFAULT_FILTER_MODEL,
    FILTER_OUTPUT_SCHEMA,
    build_filter_prompt,
    llm_filter_candidates,
)
from bioscancast.filtering.models import (
    FilterDecision,
    ForecastQuestion,
    SearchResult,
)
from bioscancast.llm.base import LLMResponse
from bioscancast.llm.fake_client import FakeLLMClient


def _make_question() -> ForecastQuestion:
    return ForecastQuestion(
        id="q1",
        text="Will US H5N1 herds exceed 1500 by June 2026?",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        pathogen="H5N1",
        region="United States",
        event_type="case_count",
    )


def _make_result(rid: str, url: str) -> SearchResult:
    return SearchResult(
        id=rid, question_id="q1", query_id="qx", engine="tavily",
        url=url, canonical_url=url, domain="cdc.gov",
        title=f"Result {rid}", snippet="data on H5N1 herds", rank=1,
        retrieved_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )


def _make_decision(rid: str) -> FilterDecision:
    return FilterDecision(
        result_id=rid, keep=None, stage="heuristic",
        relevance_score=0.5, credibility_score=0.5, priority_score=0.5,
    )


def test_build_filter_prompt_returns_triple():
    """build_filter_prompt must return (system, user, schema) matching the
    shared LLMClient.generate_json signature."""
    q = _make_question()
    system, user, schema = build_filter_prompt(q, [{"result_id": "r1"}])
    assert isinstance(system, str) and system.startswith("You are filtering")
    assert isinstance(user, str)
    # User prompt is a JSON object containing the question and candidates
    import json as _json
    payload = _json.loads(user)
    assert payload["question"]["id"] == "q1"
    assert payload["candidates"] == [{"result_id": "r1"}]
    assert schema is FILTER_OUTPUT_SCHEMA


def test_filter_output_schema_is_strict_json_schema():
    """The output schema must be a real JSON Schema (not an example)
    with strict OpenAI-compatible properties."""
    schema = FILTER_OUTPUT_SCHEMA
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    item = schema["properties"]["decisions"]["items"]
    assert item["type"] == "object"
    assert item["additionalProperties"] is False
    # All listed properties must be required
    assert set(item["required"]) == set(item["properties"].keys())


def test_llm_filter_candidates_calls_new_protocol():
    """llm_filter_candidates must call generate_json with system/user/schema/
    model/max_tokens kwargs and read from response.content (LLMResponse)."""
    q = _make_question()
    sr = _make_result("r1", "https://cdc.gov/h5n1")
    fd = _make_decision("r1")

    captured = {}

    class CapturingFake:
        def generate_json(self, *, system, user, schema, model, max_tokens):
            captured.update(
                system=system, user=user, schema=schema,
                model=model, max_tokens=max_tokens,
            )
            return LLMResponse(
                content={"decisions": [{
                    "result_id": "r1", "keep": True,
                    "relevance_score": 0.9, "credibility_score": 0.85,
                    "final_score": 0.88,
                    "reason_codes": ["official_source"],
                    "notes": "cdc dashboard",
                }]},
                input_tokens=100, output_tokens=20,
                model="gpt-4o-mini", raw_text="{}",
            )

        def embed(self, texts, *, model):
            raise NotImplementedError("filter doesn't embed")

    result = llm_filter_candidates(q, [fd], {"r1": sr}, CapturingFake())

    # Protocol call
    assert "system" in captured
    assert "user" in captured
    assert captured["schema"] is FILTER_OUTPUT_SCHEMA
    assert captured["model"] == DEFAULT_FILTER_MODEL
    assert captured["max_tokens"] == DEFAULT_FILTER_MAX_TOKENS

    # Decision parsing
    assert len(result) == 1
    decision = result[0]
    assert decision.keep is True
    assert decision.stage == "llm"
    assert decision.relevance_score == 0.9
    assert decision.credibility_score == 0.85
    assert decision.priority_score == 0.88
    assert decision.reason_codes == ["official_source"]
    assert decision.notes == "cdc dashboard"


def test_llm_filter_candidates_handles_missing_decision():
    """If the LLM omits a decision for some candidate, that candidate
    should be marked keep=False with a 'missing_llm_decision' reason."""
    q = _make_question()
    sr_a = _make_result("ra", "https://a.com")
    sr_b = _make_result("rb", "https://b.com")
    fake = FakeLLMClient([LLMResponse(
        content={"decisions": [{
            "result_id": "ra", "keep": True,
            "relevance_score": 0.8, "credibility_score": 0.8,
            "final_score": 0.8, "reason_codes": [], "notes": None,
        }]},  # rb is omitted
        input_tokens=100, output_tokens=20,
        model="gpt-4o-mini", raw_text="{}",
    )])

    result = llm_filter_candidates(
        q,
        [_make_decision("ra"), _make_decision("rb")],
        {"ra": sr_a, "rb": sr_b},
        fake,
    )

    by_id = {d.result_id: d for d in result}
    assert by_id["ra"].keep is True
    assert by_id["rb"].keep is False
    assert "missing_llm_decision" in by_id["rb"].reason_codes


def test_llm_filter_candidates_empty_input_returns_empty():
    """Empty candidate list should not call the LLM at all."""

    class ExplodingFake:
        def generate_json(self, **_):
            raise AssertionError("generate_json must not be called on empty input")
        def embed(self, *_, **__):
            raise NotImplementedError

    q = _make_question()
    assert llm_filter_candidates(q, [], {}, ExplodingFake()) == []
