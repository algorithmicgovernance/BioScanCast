"""Tests for the LLM-driven search-stage query decomposition.

Uses the shared ``bioscancast.llm.base.LLMClient`` protocol — the
``generate_json`` calls pass system/user/schema/model/max_tokens
kwargs and return ``LLMResponse`` objects.
"""

from datetime import datetime, timezone

from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.llm.base import LLMResponse
from bioscancast.stages.searching.query_decomposition import (
    CLASSIFY_SCHEMA,
    DECOMPOSE_SCHEMA,
    VALID_AXES,
    classify_question_type,
    decompose_question,
)


def _make_question(**overrides):
    defaults = {
        "id": "Q001",
        "text": "Will H5N1 cause more than 100 human cases in the US by December 2026?",
        "created_at": datetime(2025, 1, 15, tzinfo=timezone.utc),
        "pathogen": "H5N1",
        "region": "United States",
    }
    defaults.update(overrides)
    return ForecastQuestion(**defaults)


def _resp(content: dict, input_tokens: int = 80, output_tokens: int = 20) -> LLMResponse:
    """Build a minimal LLMResponse for tests."""
    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model="gpt-4o-mini",
        raw_text="{}",
    )


class FakeLLMClient:
    """Mock LLM client implementing the shared
    ``bioscancast.llm.base.LLMClient`` protocol — FIFO scripted responses
    keyed by call order, not by content."""

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self._call_count = 0
        self.recorded_calls: list[dict] = []

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.recorded_calls.append({
            "system": system, "user": user, "schema": schema,
            "model": model, "max_tokens": max_tokens,
        })
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        self._call_count += 1
        return _resp({})

    def embed(self, texts, *, model):
        raise NotImplementedError("query decomposition doesn't embed")


class TestClassifyQuestionType:
    def test_returns_outbreak_count(self):
        llm = FakeLLMClient([_resp({"question_type": "outbreak_count"})])
        assert classify_question_type(_make_question(), llm) == "outbreak_count"

    def test_returns_binary_event(self):
        llm = FakeLLMClient([_resp({"question_type": "binary_event"})])
        assert classify_question_type(_make_question(), llm) == "binary_event"

    def test_invalid_type_falls_back_to_unknown(self):
        llm = FakeLLMClient([_resp({"question_type": "nonsense"})])
        assert classify_question_type(_make_question(), llm) == "unknown"

    def test_llm_failure_falls_back_to_unknown(self):
        class FailingLLM:
            def generate_json(self, **_):
                raise RuntimeError("LLM down")
            def embed(self, *_, **__):
                raise NotImplementedError

        assert classify_question_type(_make_question(), FailingLLM()) == "unknown"

    def test_calls_new_protocol_with_classify_schema(self):
        llm = FakeLLMClient([_resp({"question_type": "outbreak_count"})])
        classify_question_type(_make_question(), llm)
        assert len(llm.recorded_calls) == 1
        call = llm.recorded_calls[0]
        assert call["schema"] is CLASSIFY_SCHEMA
        assert isinstance(call["system"], str) and "biosecurity" in call["system"]
        assert isinstance(call["user"], str)


class TestDecomposeQuestion:
    def test_produces_valid_subqueries(self):
        classify_resp = _resp({"question_type": "outbreak_count"})
        decompose_resp = _resp({
            "sub_queries": [
                {"text": "H5N1 human cases US 2025", "axis": "latest_data"},
                {"text": "H5N1 outbreak trend growth", "axis": "trend"},
                {"text": "avian influenza government response", "axis": "policy"},
                {"text": "H5N1 historical human cases", "axis": "historical_analogy"},
                {"text": "bird flu cases latest report", "axis": "latest_data"},
            ]
        })
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)

        assert len(result) >= 3
        assert len(result) <= 8
        for sq in result:
            assert sq.axis in VALID_AXES
            assert sq.question_id == "Q001"
            assert len(sq.id) > 0
            word_count = len(sq.text.split())
            assert 2 <= word_count <= 8

    def test_invalid_axis_dropped(self):
        classify_resp = _resp({"question_type": "unknown"})
        decompose_resp = _resp({
            "sub_queries": [
                {"text": "H5N1 cases latest", "axis": "latest_data"},
                {"text": "some bad axis query", "axis": "invalid_axis"},
                {"text": "bird flu trend analysis", "axis": "trend"},
                {"text": "avian influenza policy response", "axis": "policy"},
            ]
        })
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)

        axes = [sq.axis for sq in result]
        assert "invalid_axis" not in axes

    def test_too_long_query_truncated(self):
        classify_resp = _resp({"question_type": "unknown"})
        decompose_resp = _resp({
            "sub_queries": [
                {"text": "one two three four five six seven eight nine ten", "axis": "latest_data"},
                {"text": "H5N1 trend data", "axis": "trend"},
                {"text": "avian flu policy update", "axis": "policy"},
                {"text": "bird flu expert analysis view", "axis": "expert_opinion"},
                {"text": "historical outbreak comparison data", "axis": "historical_analogy"},
            ]
        })
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)

        for sq in result:
            assert len(sq.text.split()) <= 8

    def test_llm_failure_uses_fallback(self):
        class FailingLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, **_):
                self.calls += 1
                if self.calls == 1:
                    return _resp({"question_type": "unknown"})
                raise RuntimeError("LLM decomposition failed")

            def embed(self, *_, **__):
                raise NotImplementedError

        result = decompose_question(_make_question(), FailingLLM())
        assert len(result) >= 1
        for sq in result:
            assert sq.axis in VALID_AXES

    def test_malformed_response_uses_fallback(self):
        classify_resp = _resp({"question_type": "unknown"})
        decompose_resp = _resp({"sub_queries": "not a list"})
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)
        assert len(result) >= 1

    def test_caps_at_8(self):
        classify_resp = _resp({"question_type": "unknown"})
        decompose_resp = _resp({
            "sub_queries": [
                {"text": f"query number {i} text", "axis": "latest_data"}
                for i in range(12)
            ]
        })
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)
        assert len(result) <= 8

    def test_uses_decompose_schema_on_second_call(self):
        classify_resp = _resp({"question_type": "outbreak_count"})
        decompose_resp = _resp({"sub_queries": [
            {"text": "H5N1 cases US 2025", "axis": "latest_data"},
            {"text": "avian flu trend", "axis": "trend"},
            {"text": "USDA policy avian", "axis": "policy"},
        ]})
        llm = FakeLLMClient([classify_resp, decompose_resp])
        decompose_question(_make_question(), llm)
        assert len(llm.recorded_calls) == 2
        assert llm.recorded_calls[0]["schema"] is CLASSIFY_SCHEMA
        assert llm.recorded_calls[1]["schema"] is DECOMPOSE_SCHEMA
