from __future__ import annotations

import json

import bioscancast.main as orchestrator
from bioscancast.main import _build_forecast_history_context
from bioscancast.stages.forecasting.schemas import (
    ForecastDistribution,
    ForecastRecord,
    ForecastResult,
)
from bioscancast.stages.insight.pipeline import InsightRunResult


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_forecast_history_context_reads_prior_runs(tmp_path):
    qid = "q-hist"
    current = "20260714_120000"

    old_run = tmp_path / qid / "20260701_000000"
    _write_json(
        old_run / "question.json",
        {"id": qid, "as_of_date": "2026-07-01T00:00:00+00:00"},
    )
    _write_json(
        old_run / "forecast.json",
        {
            "distributions": [
                {
                    "forecast_source": "bioscancast",
                    "probabilities": {"YES": 0.72, "NO": 0.28},
                }
            ],
            "samples": [
                {"ok": True, "rationale": "cases rose over prior two weeks"}
            ],
            "baseline_rationale": "prior-only baseline",
        },
    )

    context = _build_forecast_history_context(tmp_path, qid, current)

    assert "as_of=2026-07-01T00:00:00+00:00" in context
    assert "top=YES (0.720)" in context
    assert "rationale: cases rose over prior two weeks" in context


def test_build_forecast_history_context_skips_current_run(tmp_path):
    qid = "q-hist"
    current = "20260714_120000"

    cur_run = tmp_path / qid / current
    _write_json(cur_run / "question.json", {"id": qid, "as_of_date": "2026-07-14"})
    _write_json(
        cur_run / "forecast.json",
        {
            "distributions": [
                {
                    "forecast_source": "bioscancast",
                    "probabilities": {"YES": 0.99, "NO": 0.01},
                }
            ],
            "samples": [{"ok": True, "rationale": "current run"}],
        },
    )

    context = _build_forecast_history_context(tmp_path, qid, current)
    assert context == ""


def test_orchestrator_passes_history_block_from_prior_runs(tmp_path, monkeypatch):
    qid = "q-int"
    current = "20260714_120000"

    # Two prior runs for the same question.
    for run_id, as_of, probs, rationale in (
        ("20260712_000000", "2026-07-12T00:00:00+00:00", {"YES": 0.7, "NO": 0.3}, "uptick continued"),
        ("20260710_000000", "2026-07-10T00:00:00+00:00", {"YES": 0.6, "NO": 0.4}, "signals mixed"),
    ):
        rdir = tmp_path / qid / run_id
        _write_json(rdir / "question.json", {"id": qid, "as_of_date": as_of})
        _write_json(
            rdir / "forecast.json",
            {
                "distributions": [
                    {
                        "forecast_source": "bioscancast",
                        "probabilities": probs,
                    }
                ],
                "samples": [{"ok": True, "rationale": rationale}],
            },
        )

    # Unrelated question history must not be included.
    other = tmp_path / "q-other" / "20260711_000000"
    _write_json(other / "question.json", {"id": "q-other", "as_of_date": "2026-07-11"})
    _write_json(
        other / "forecast.json",
        {
            "distributions": [
                {
                    "forecast_source": "bioscancast",
                    "probabilities": {"YES": 0.99, "NO": 0.01},
                }
            ],
            "samples": [{"ok": True, "rationale": "other question"}],
        },
    )

    captured: dict[str, str | None] = {"history": None}

    class _DummyLLM:
        def generate_json(self, **kwargs):  # pragma: no cover
            raise AssertionError("LLM should not be called in this stubbed test")

        def embed(self, texts, *, model):
            return [[0.0] * 4 for _ in texts]

    class _DummyBackend:
        pass

    class _SearchStagePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, question):
            return []

    class _FilteringPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, question, search_results):
            return []

    class _ExtractionPipeline:
        def __init__(self, **kwargs):
            self.docling_telemetry = []

        def run(self, filtered_docs):
            return []

    class _InsightPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, question, documents):
            return InsightRunResult(
                records=[],
                budget_summary={
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "per_model": {},
                },
                documents_processed=0,
                documents_skipped=0,
                notes=[],
            )

    class _ForecastingPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, question, records, options, historical_context=None):
            captured["history"] = historical_context
            return ForecastResult(
                question_id=question.id,
                options=list(options),
                distributions=[
                    ForecastDistribution(
                        question_id=question.id,
                        forecast_source="bioscancast",
                        probabilities={"YES": 0.5, "NO": 0.5},
                    )
                ],
                records=[
                    ForecastRecord(
                        question_id=question.id,
                        forecast_source="bioscancast",
                        option="YES",
                        probability=0.5,
                    ),
                    ForecastRecord(
                        question_id=question.id,
                        forecast_source="bioscancast",
                        option="NO",
                        probability=0.5,
                    ),
                ],
                budget_summary={
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "per_model": {},
                },
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(orchestrator, "OpenAILLMClient", _DummyLLM)
    monkeypatch.setattr(orchestrator, "TavilyBackend", _DummyBackend)
    monkeypatch.setattr(orchestrator, "SearchStagePipeline", _SearchStagePipeline)
    monkeypatch.setattr(orchestrator, "FilteringPipeline", _FilteringPipeline)
    monkeypatch.setattr(orchestrator, "ExtractionPipeline", _ExtractionPipeline)
    monkeypatch.setattr(orchestrator, "InsightPipeline", _InsightPipeline)
    monkeypatch.setattr(orchestrator, "ForecastingPipeline", _ForecastingPipeline)

    args = orchestrator._parse_args(
        [
            qid,
            "--question",
            "Will cases exceed threshold?",
            "--out-root",
            str(tmp_path),
            "--run-id",
            current,
            "--options",
            "YES,NO",
            "--no-baseline",
            "--no-cache",
        ]
    )

    orchestrator.run_pipeline(args)

    history = captured["history"] or ""
    assert "as_of=2026-07-12T00:00:00+00:00" in history
    assert "as_of=2026-07-10T00:00:00+00:00" in history
    assert "rationale: uptick continued" in history
    assert "rationale: signals mixed" in history
    assert "other question" not in history
