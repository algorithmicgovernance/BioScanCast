from datetime import datetime, timezone

from bioscancast.stages.filtering.models import ForecastQuestion, SearchResult
from bioscancast.llm.base import LLMResponse
from bioscancast.stages.evaluation.contamination import (
    BaselineForecast,
    ContaminationCounts,
    filter_caught_contamination_rate,
    retrieval_free_baseline_forecast,
)


def _llm_response(content: dict) -> LLMResponse:
    return LLMResponse(
        content=content,
        input_tokens=0,
        output_tokens=0,
        model="fake",
        raw_text="",
    )


def _result(pub: datetime | None) -> SearchResult:
    return SearchResult(
        id="x",
        question_id="Q",
        query_id="q1",
        engine="fake",
        url="https://example.com/x",
        canonical_url="https://example.com/x",
        domain="example.com",
        title="T",
        snippet="S",
        rank=1,
        retrieved_at=datetime.now(timezone.utc),
        published_date=pub,
    )


class TestFilterCaughtContaminationRate:
    def test_clean_run_is_zero(self):
        cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
        results = [
            _result(datetime(2024, 1, 1, tzinfo=timezone.utc)),
            _result(datetime(2024, 5, 31, tzinfo=timezone.utc)),
        ]
        counts = filter_caught_contamination_rate(results, cutoff)
        assert counts.post_cutoff_in_final == 0
        assert counts.filter_caught_rate == 0.0
        assert counts.pre_cutoff_in_final == 2

    def test_some_leak_through(self):
        cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
        results = [
            _result(datetime(2024, 5, 1, tzinfo=timezone.utc)),
            _result(datetime(2024, 8, 1, tzinfo=timezone.utc)),  # post
            _result(datetime(2024, 9, 1, tzinfo=timezone.utc)),  # post
            _result(None),  # undated
        ]
        counts = filter_caught_contamination_rate(results, cutoff)
        assert counts.post_cutoff_in_final == 2
        assert counts.undated_in_final == 1
        assert counts.pre_cutoff_in_final == 1
        assert counts.filter_caught_rate == 0.5

    def test_empty_list(self):
        counts = filter_caught_contamination_rate(
            [], datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        assert counts.filter_caught_rate == 0.0
        assert counts.total == 0


class TestRetrievalFreeBaselineForecast:
    def test_well_formed_response(self):
        question = ForecastQuestion(
            id="Q1",
            text="Will X happen?",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            as_of_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

        class GoodLLM:
            def generate_json(self, *, system, user, schema, model, max_tokens=1024):
                return _llm_response({"probabilities": [0.7, 0.3], "rationale": "guess"})

        out = retrieval_free_baseline_forecast(
            question, options=["yes", "no"], llm_client=GoodLLM()
        )
        assert isinstance(out, BaselineForecast)
        assert abs(sum(out.probabilities) - 1.0) < 1e-9
        assert out.probabilities[0] > out.probabilities[1]
        assert out.rationale == "guess"

    def test_renormalises_unnormalised_probabilities(self):
        question = ForecastQuestion(
            id="Q1",
            text="?",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        class UnnormLLM:
            def generate_json(self, *, system, user, schema, model, max_tokens=1024):
                return _llm_response({"probabilities": [2.0, 6.0], "rationale": ""})

        out = retrieval_free_baseline_forecast(
            question, options=["a", "b"], llm_client=UnnormLLM()
        )
        assert abs(sum(out.probabilities) - 1.0) < 1e-9
        assert abs(out.probabilities[0] - 0.25) < 1e-9

    def test_malformed_response_uniform_fallback(self):
        question = ForecastQuestion(
            id="Q1",
            text="?",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        class BadLLM:
            def generate_json(self, *, system, user, schema, model, max_tokens=1024):
                return _llm_response({"probabilities": "not a list"})

        out = retrieval_free_baseline_forecast(
            question, options=["a", "b", "c"], llm_client=BadLLM()
        )
        assert out.probabilities == [1 / 3, 1 / 3, 1 / 3]
        assert "fallback" in (out.rationale or "")

    def test_llm_exception_uniform_fallback(self):
        question = ForecastQuestion(
            id="Q1",
            text="?",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        class ExplodingLLM:
            def generate_json(self, *, system, user, schema, model, max_tokens=1024):
                raise RuntimeError("oops")

        out = retrieval_free_baseline_forecast(
            question, options=["a", "b"], llm_client=ExplodingLLM()
        )
        assert out.probabilities == [0.5, 0.5]
