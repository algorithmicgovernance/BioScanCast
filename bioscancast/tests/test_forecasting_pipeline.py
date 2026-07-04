"""Tests for the forecasting stage.

Covers the pure-function aggregation and evidence helpers, then the full
ForecastingPipeline driven by a scripted FakeLLMClient (no network). Ends
with an integration check that the output feeds the eval stage's scorer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.stages.forecasting import aggregation
from bioscancast.stages.forecasting.config import ForecastingConfig
from bioscancast.stages.forecasting.evidence import build_evidence_digest, _format_record
from bioscancast.stages.forecasting.pipeline import ForecastingPipeline
from bioscancast.stages.forecasting.schemas import ForecastResult
from bioscancast.llm.base import LLMResponse
from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.schemas import ChunkReference, InsightRecord
from bioscancast.stages.evaluation.scoring import (
    log_score,
    multiclass_brier_score,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


QUESTION = ForecastQuestion(
    id="q-test",
    text="Will there be more than 100 cases by June 2026?",
    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    target_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
    region="Uganda",
    pathogen="Sudan virus",
    resolution_criteria="YES if confirmed cases exceed 100.",
)


def _record(
    *,
    rid: str = "ins-1",
    metric_value=9.0,
    event_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
    precision="day",
    confidence=0.9,
    summary=None,
    quote="9 confirmed cases",
    count_basis=None,
    time_window=None,
    data_quality=None,
) -> InsightRecord:
    return InsightRecord(
        id=rid,
        question_id="q-test",
        event_type="case_count",
        confidence=confidence,
        location="Uganda",
        iso_country_code="UG",
        pathogen="Sudan virus",
        metric_name="confirmed_cases",
        metric_value=metric_value,
        metric_unit="cases",
        count_basis=count_basis,
        time_window=time_window,
        data_quality=data_quality,
        event_date=event_date,
        event_date_precision=precision,
        summary=summary,
        sources=[
            ChunkReference(
                document_id="d1",
                chunk_id="c1",
                source_url="https://who.int/sitrep",
                quote=quote,
            )
        ],
    )


def _forecast_resp(
    probabilities,
    *,
    model="gpt-4o",
    input_tokens=300,
    output_tokens=120,
) -> LLMResponse:
    """A well-formed ensemble-sample response."""
    content = {
        "reference_class": "historical filovirus outbreaks",
        "base_rate": 0.25,
        "drivers_up": ["rising case count", "porous borders"],
        "drivers_down": ["rapid vaccination", "containment"],
        "why_might_be_wrong": "reporting lag could hide cases",
        "rationale": "outside view adjusted up modestly",
        "probabilities": probabilities,
    }
    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        raw_text="{}",
    )


def _baseline_resp(probabilities, *, model="gpt-4o-mini") -> LLMResponse:
    return LLMResponse(
        content={"probabilities": probabilities, "rationale": "prior only"},
        input_tokens=50,
        output_tokens=20,
        model=model,
        raw_text="{}",
    )


def _malformed_resp() -> LLMResponse:
    """Mimics the real client's content={} on unparseable JSON."""
    return LLMResponse(
        content={}, input_tokens=300, output_tokens=0, model="gpt-4o", raw_text=""
    )


# ---------------------------------------------------------------------------
# Aggregation unit tests
# ---------------------------------------------------------------------------


def test_geometric_mean_of_odds_matches_hand_calc():
    # odds 9, 1.5, 2.333 -> product 31.5 -> ^(1/3) = 3.1583 -> p = 0.7595
    result = aggregation.geometric_mean_of_odds(
        [[0.9, 0.1], [0.6, 0.4], [0.7, 0.3]]
    )
    assert result[0] == pytest.approx(0.7595, abs=1e-3)
    assert sum(result) == pytest.approx(1.0)


def test_geometric_mean_of_odds_preserves_confidence():
    # Log-odds pooling is *more* confident than the arithmetic mean when
    # members agree in direction: it sits further from 0.5. This is the
    # property superforecasting favours (linear averaging dilutes toward
    # 0.5).
    samples = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    pooled = aggregation.geometric_mean_of_odds(samples)
    mean = aggregation.arithmetic_mean(samples)
    assert pooled[0] > mean[0]
    assert sum(pooled) == pytest.approx(1.0)


def test_mean_and_median():
    assert aggregation.arithmetic_mean([[0.8, 0.2], [0.4, 0.6]]) == pytest.approx(
        [0.6, 0.4]
    )
    med = aggregation.median([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4]])
    assert med == pytest.approx([0.6, 0.4])


def test_extremize_off_is_noop_and_on_sharpens():
    assert aggregation.extremize([0.6, 0.4], 0.0) == pytest.approx([0.6, 0.4])
    sharper = aggregation.extremize([0.6, 0.4], 1.0)
    assert sharper[0] > 0.6
    assert sum(sharper) == pytest.approx(1.0)


def test_aggregate_extremize_gate():
    # Pooled peak decides whether the gate lets extremizing through.
    peaked = [[0.9, 0.1], [0.85, 0.15]]   # mean -> peak 0.875
    diffuse = [[0.5, 0.5], [0.55, 0.45]]  # mean -> peak 0.525

    pooled_peaked = aggregation.aggregate(peaked, "mean")
    pooled_diffuse = aggregation.aggregate(diffuse, "mean")

    # Gate 0.6: peaked pool clears it and is sharpened...
    gated_peaked = aggregation.aggregate(
        peaked, "mean", extremize_strength=1.0, extremize_gate=0.6
    )
    assert gated_peaked[0] > pooled_peaked[0]
    # ...diffuse pool does not clear the gate and is left untouched.
    gated_diffuse = aggregation.aggregate(
        diffuse, "mean", extremize_strength=1.0, extremize_gate=0.6
    )
    assert gated_diffuse == pytest.approx(pooled_diffuse)

    # No gate (default): extremizing applies regardless of peak.
    ungated_diffuse = aggregation.aggregate(
        diffuse, "mean", extremize_strength=1.0, extremize_gate=None
    )
    assert ungated_diffuse[0] > pooled_diffuse[0]


def test_aggregate_validations():
    with pytest.raises(ValueError):
        aggregation.aggregate([], "mean")
    with pytest.raises(ValueError):
        aggregation.aggregate([[0.5, 0.5]], "not_a_method")
    with pytest.raises(ValueError):
        aggregation.aggregate([[0.5, 0.5], [0.3, 0.3, 0.4]], "mean")


# ---------------------------------------------------------------------------
# Evidence digest unit tests
# ---------------------------------------------------------------------------


def test_evidence_digest_empty_and_contentless():
    assert build_evidence_digest([]) == ("", [])
    # A record with neither metric nor summary is noise -> skipped.
    blank = _record(rid="blank", metric_value=None, summary=None)
    assert build_evidence_digest([blank]) == ("", [])


def test_evidence_digest_recency_sorted_and_capped():
    older = _record(
        rid="old", metric_value=5, event_date=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    newer = _record(
        rid="new", metric_value=20, event_date=datetime(2026, 3, 1, tzinfo=timezone.utc)
    )
    digest, ids = build_evidence_digest([older, newer], max_records=5)
    # Newer first.
    assert ids == ["new", "old"]
    assert digest.index("2026-03") < digest.index("2026-01")

    capped, capped_ids = build_evidence_digest([older, newer], max_records=1)
    assert capped_ids == ["new"]
    assert capped.count("\n") == 0  # single line


def test_evidence_digest_line_format():
    digest, _ = build_evidence_digest([_record()])
    assert "[2026-01-15]" in digest
    assert "Sudan virus" in digest
    assert "confirmed_cases=9 cases" in digest
    assert '"9 confirmed cases"' in digest
    assert "https://who.int/sitrep" in digest


def test_evidence_digest_surfaces_count_basis():
    """The digest must tag metric lines with their count basis so the
    forecaster can tell cumulative / incident / active numbers apart. A
    plain metric line (no basis, or 'unknown') stays untagged."""
    cumulative = _record(rid="cum", count_basis="cumulative")
    incident = _record(rid="inc", count_basis="incident", time_window="week")
    active = _record(rid="act", count_basis="active")
    unknown = _record(rid="unk", count_basis="unknown")
    plain = _record(rid="plain")  # count_basis=None

    lines = {
        r.id: _format_record(r)
        for r in (cumulative, incident, active, unknown, plain)
    }
    assert "confirmed_cases=9 cases [cumulative]" in lines["cum"]
    # incident carries its reporting window.
    assert "confirmed_cases=9 cases [incident/week]" in lines["inc"]
    assert "confirmed_cases=9 cases [active]" in lines["act"]
    # "unknown" and None add no basis tag (keep the line clean).
    assert "confirmed_cases=9 cases [" not in lines["unk"]
    assert "confirmed_cases=9 cases [" not in lines["plain"]


def test_evidence_digest_surfaces_data_quality_on_metric_line():
    """An explicit data-quality caveat must appear in the digest even on a
    metric-bearing record, whose free-text summary is otherwise dropped."""
    rec = _record(
        rid="dq",
        count_basis="cumulative",
        summary="cumulative total since outbreak start",  # dropped for metric lines
        data_quality="testing limited; many mild cases not captured",
    )
    line = _format_record(rec)
    assert "confirmed_cases=9 cases [cumulative]" in line
    assert "(caveat: testing limited; many mild cases not captured)" in line
    # The summary itself is still not rendered on a metric line.
    assert "cumulative total since outbreak start" not in line


def test_evidence_digest_no_caveat_when_data_quality_absent():
    assert "(caveat:" not in _format_record(_record())


# ---------------------------------------------------------------------------
# Pipeline tests (binary)
# ---------------------------------------------------------------------------


def _config(**overrides) -> ForecastingConfig:
    base = dict(ensemble_samples=3, temperature=0.0)
    base.update(overrides)
    return ForecastingConfig(**base)


def test_pipeline_binary_ensemble_and_baseline():
    client = FakeLLMClient([
        _forecast_resp([0.7, 0.3]),
        _forecast_resp([0.6, 0.4]),
        _forecast_resp([0.8, 0.2]),
        _baseline_resp([0.4, 0.6]),
    ])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())

    result = pipeline.run(QUESTION, [_record()], ["YES", "NO"])

    assert isinstance(result, ForecastResult)
    # Two sources: evidence-based + baseline.
    sources = {d.forecast_source for d in result.distributions}
    assert sources == {"bioscancast", "bioscancast_baseline"}

    primary = next(d for d in result.distributions if d.forecast_source == "bioscancast")
    assert set(primary.probabilities) == {"YES", "NO"}
    assert sum(primary.probabilities.values()) == pytest.approx(1.0)
    # Geometric-mean-of-odds pool of three YES-leaning samples stays > 0.5.
    assert primary.probabilities["YES"] > 0.5

    # Flat records: one row per option per source = 2 sources * 2 options.
    assert len(result.records) == 4
    for rec in result.records:
        assert rec.question_id == "q-test"
        assert 0.0 <= rec.probability <= 1.0

    # Reasoning captured for audit.
    assert len(result.samples) == 3
    assert all(s.ok for s in result.samples)
    assert result.samples[0].reference_class
    assert result.samples[0].drivers_up

    # Baseline's rationale is captured too (for informed-vs-baseline audit).
    assert result.baseline_rationale == "prior only"

    # Distinct seeds per sample.
    assert [s.seed for s in result.samples] == [42, 43, 44]

    # Evidence + budget tracked (ensemble + baseline tokens).
    assert result.evidence_record_ids == ["ins-1"]
    assert result.budget_summary["total_input_tokens"] > 0
    assert result.budget_summary["total_output_tokens"] > 0


def test_pipeline_tracks_ensemble_and_baseline_tokens():
    # The ensemble runs on gpt-4o and the baseline on gpt-4o-mini (through
    # the budget-recording wrapper). The budget must capture BOTH, per
    # model, not just the ensemble.
    client = FakeLLMClient([
        _forecast_resp([0.7, 0.3], input_tokens=300, output_tokens=120),
        _forecast_resp([0.6, 0.4], input_tokens=300, output_tokens=120),
        _forecast_resp([0.8, 0.2], input_tokens=300, output_tokens=120),
        _baseline_resp([0.4, 0.6]),  # gpt-4o-mini, input 50 / output 20
    ])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())
    result = pipeline.run(QUESTION, [_record()], ["YES", "NO"])

    per_model = result.budget_summary["per_model"]
    assert per_model["gpt-4o"]["calls"] == 3          # ensemble
    assert per_model["gpt-4o-mini"]["calls"] == 1      # baseline
    # Totals include the baseline's tokens, not just the ensemble's.
    assert result.budget_summary["total_input_tokens"] == 300 * 3 + 50
    assert result.budget_summary["total_output_tokens"] == 120 * 3 + 20


def test_pipeline_no_baseline():
    client = FakeLLMClient([
        _forecast_resp([0.7, 0.3]),
        _forecast_resp([0.6, 0.4]),
        _forecast_resp([0.8, 0.2]),
    ])
    pipeline = ForecastingPipeline(
        llm_client=client, config=_config(emit_baseline=False)
    )
    result = pipeline.run(QUESTION, [_record()], ["YES", "NO"])
    assert [d.forecast_source for d in result.distributions] == ["bioscancast"]
    assert len(result.records) == 2


# ---------------------------------------------------------------------------
# Pipeline tests (range / multiclass)
# ---------------------------------------------------------------------------


def test_pipeline_range_multiclass():
    options = ["70-100", "100-150", "150-200", "200+"]
    client = FakeLLMClient([
        _forecast_resp([0.6, 0.2, 0.1, 0.1]),
        _forecast_resp([0.5, 0.3, 0.1, 0.1]),
        _forecast_resp([0.7, 0.2, 0.05, 0.05]),
        _baseline_resp([0.25, 0.25, 0.25, 0.25]),
    ])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())
    result = pipeline.run(QUESTION, [_record()], options)

    primary = next(d for d in result.distributions if d.forecast_source == "bioscancast")
    assert set(primary.probabilities) == set(options)
    assert sum(primary.probabilities.values()) == pytest.approx(1.0)
    assert max(primary.probabilities, key=primary.probabilities.get) == "70-100"


# ---------------------------------------------------------------------------
# Robustness / fallbacks
# ---------------------------------------------------------------------------


def test_pipeline_skips_malformed_sample():
    # Middle sample is malformed (content={}, wrong shape) -> dropped, the
    # other two still aggregate.
    client = FakeLLMClient([
        _forecast_resp([0.7, 0.3]),
        _malformed_resp(),
        _forecast_resp([0.8, 0.2]),
        _baseline_resp([0.5, 0.5]),
    ])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())
    result = pipeline.run(QUESTION, [_record()], ["YES", "NO"])

    ok = [s for s in result.samples if s.ok]
    assert len(result.samples) == 3
    assert len(ok) == 2
    primary = next(d for d in result.distributions if d.forecast_source == "bioscancast")
    assert sum(primary.probabilities.values()) == pytest.approx(1.0)


def test_pipeline_all_samples_fail_uniform_fallback():
    client = FakeLLMClient([
        _malformed_resp(),
        _malformed_resp(),
        _malformed_resp(),
        _baseline_resp([0.4, 0.6]),
    ])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())
    result = pipeline.run(QUESTION, [_record()], ["YES", "NO"])

    primary = next(d for d in result.distributions if d.forecast_source == "bioscancast")
    assert primary.probabilities["YES"] == pytest.approx(0.5)
    assert primary.probabilities["NO"] == pytest.approx(0.5)
    assert any("uniform" in n.lower() for n in result.notes)


def test_pipeline_no_evidence_notes_priors():
    client = FakeLLMClient([
        _forecast_resp([0.3, 0.7]),
        _forecast_resp([0.4, 0.6]),
        _forecast_resp([0.35, 0.65]),
        _baseline_resp([0.5, 0.5]),
    ])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())
    # No records -> empty digest.
    result = pipeline.run(QUESTION, [], ["YES", "NO"])
    assert result.evidence_record_ids == []
    assert any("prior" in n.lower() for n in result.notes)
    # Still produces a usable primary forecast.
    primary = next(d for d in result.distributions if d.forecast_source == "bioscancast")
    assert sum(primary.probabilities.values()) == pytest.approx(1.0)


def test_pipeline_rejects_empty_options():
    client = FakeLLMClient([])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())
    with pytest.raises(ValueError):
        pipeline.run(QUESTION, [_record()], [])


# ---------------------------------------------------------------------------
# Integration with the eval stage scorer
# ---------------------------------------------------------------------------


def test_output_feeds_eval_scorer():
    options = ["YES", "NO"]
    client = FakeLLMClient([
        _forecast_resp([0.7, 0.3]),
        _forecast_resp([0.6, 0.4]),
        _forecast_resp([0.8, 0.2]),
        _baseline_resp([0.5, 0.5]),
    ])
    pipeline = ForecastingPipeline(llm_client=client, config=_config())
    result = pipeline.run(QUESTION, [_record()], options)

    primary = next(d for d in result.distributions if d.forecast_source == "bioscancast")
    probs = [primary.probabilities[o] for o in options]
    # Resolved YES (index 0): scorer returns finite scores.
    brier = multiclass_brier_score(probs, true_index=0)
    logs = log_score(probs, true_index=0)
    assert 0.0 <= brier <= 2.0
    assert logs >= 0.0
