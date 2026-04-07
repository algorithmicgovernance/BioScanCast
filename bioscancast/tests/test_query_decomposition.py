from datetime import datetime, timezone

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.stages.search_stage.query_decomposition import (
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


class FakeLLMClient:
    """Mock LLM client that returns canned responses."""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    def generate_json(self, prompt: str) -> dict:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        self._call_count += 1
        return {}


class TestClassifyQuestionType:
    def test_returns_outbreak_count(self):
        llm = FakeLLMClient([{"question_type": "outbreak_count"}])
        assert classify_question_type(_make_question(), llm) == "outbreak_count"

    def test_returns_binary_event(self):
        llm = FakeLLMClient([{"question_type": "binary_event"}])
        assert classify_question_type(_make_question(), llm) == "binary_event"

    def test_invalid_type_falls_back_to_unknown(self):
        llm = FakeLLMClient([{"question_type": "nonsense"}])
        assert classify_question_type(_make_question(), llm) == "unknown"

    def test_llm_failure_falls_back_to_unknown(self):
        class FailingLLM:
            def generate_json(self, prompt: str) -> dict:
                raise RuntimeError("LLM down")

        assert classify_question_type(_make_question(), FailingLLM()) == "unknown"


class TestDecomposeQuestion:
    def test_produces_valid_subqueries(self):
        classify_resp = {"question_type": "outbreak_count"}
        decompose_resp = {
            "sub_queries": [
                {"text": "H5N1 human cases US 2025", "axis": "latest_data"},
                {"text": "H5N1 outbreak trend growth", "axis": "trend"},
                {"text": "avian influenza government response", "axis": "policy"},
                {"text": "H5N1 historical human cases", "axis": "historical_analogy"},
                {"text": "bird flu cases latest report", "axis": "latest_data"},
            ]
        }
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
        classify_resp = {"question_type": "unknown"}
        decompose_resp = {
            "sub_queries": [
                {"text": "H5N1 cases latest", "axis": "latest_data"},
                {"text": "some bad axis query", "axis": "invalid_axis"},
                {"text": "bird flu trend analysis", "axis": "trend"},
                {"text": "avian influenza policy response", "axis": "policy"},
            ]
        }
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)

        axes = [sq.axis for sq in result]
        assert "invalid_axis" not in axes

    def test_too_long_query_truncated(self):
        classify_resp = {"question_type": "unknown"}
        decompose_resp = {
            "sub_queries": [
                {"text": "one two three four five six seven eight nine ten", "axis": "latest_data"},
                {"text": "H5N1 trend data", "axis": "trend"},
                {"text": "avian flu policy update", "axis": "policy"},
                {"text": "bird flu expert analysis view", "axis": "expert_opinion"},
                {"text": "historical outbreak comparison data", "axis": "historical_analogy"},
            ]
        }
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)

        for sq in result:
            assert len(sq.text.split()) <= 8

    def test_llm_failure_uses_fallback(self):
        class FailingLLM:
            def __init__(self):
                self.calls = 0

            def generate_json(self, prompt: str) -> dict:
                self.calls += 1
                if self.calls == 1:
                    return {"question_type": "unknown"}
                raise RuntimeError("LLM decomposition failed")

        result = decompose_question(_make_question(), FailingLLM())
        assert len(result) >= 1
        for sq in result:
            assert sq.axis in VALID_AXES

    def test_malformed_response_uses_fallback(self):
        classify_resp = {"question_type": "unknown"}
        decompose_resp = {"sub_queries": "not a list"}
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)
        assert len(result) >= 1

    def test_caps_at_8(self):
        classify_resp = {"question_type": "unknown"}
        decompose_resp = {
            "sub_queries": [
                {"text": f"query number {i} text", "axis": "latest_data"}
                for i in range(12)
            ]
        }
        llm = FakeLLMClient([classify_resp, decompose_resp])
        result = decompose_question(_make_question(), llm)
        assert len(result) <= 8
