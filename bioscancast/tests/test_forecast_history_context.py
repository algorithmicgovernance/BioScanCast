from __future__ import annotations

import json

from bioscancast.main import _build_forecast_history_context


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
