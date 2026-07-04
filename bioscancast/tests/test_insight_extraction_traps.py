"""Tests for the insight extraction trap-set scorer + a live diagnostic.

Two layers:

1. **Pure-scorer unit tests** (no network) validate that
   ``score_extraction`` correctly flags each mishandling mode and is
   forward-compatible with a future ``value_basis`` field. These prove the
   instrument is sound without needing API keys.

2. **Integration through ``extract_facts_from_chunk``** with a scripted
   ``FakeLLMClient`` confirms a "bad" model response (projection filed as an
   observed count) is caught and a "good" one passes.

3. **A ``@pytest.mark.live`` diagnostic** runs the real extractor over the
   whole trap-set and prints a report. Opt in with ``--live`` and a real
   ``OPENAI_API_KEY``; choose the model with ``--trap-model`` (default
   ``gpt-4o-mini``). This is the A2 diagnosis instrument.
"""

from __future__ import annotations

import os

import pytest

from bioscancast.llm.base import LLMResponse
from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.schemas import InsightRecord
from bioscancast.stages.insight.text_extraction.chunk_extractor import extract_facts_from_chunk

from bioscancast.tests.fixtures.insight.extraction_traps import (
    ALL_TRAPS,
    TRAP_PROJECTION,
    aggregate,
    score_extraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(**kwargs) -> InsightRecord:
    """Build an InsightRecord with sensible defaults for scorer tests."""
    base = dict(
        id="ins-test",
        question_id="q-test",
        event_type="case_count",
        confidence=1.0,
    )
    base.update(kwargs)
    return InsightRecord(**base)


def _fact(**kwargs) -> dict:
    """Build a fact dict matching the extraction schema."""
    fact = {
        "event_type": "case_count",
        "confidence": 1.0,
        "location": None,
        "pathogen": None,
        "metric_name": None,
        "metric_value": None,
        "metric_unit": None,
        "event_date": None,
        "summary": None,
        "quote": "",
    }
    fact.update(kwargs)
    return fact


def _response(facts: list[dict]) -> LLMResponse:
    return LLMResponse(
        content={"facts": facts},
        input_tokens=200,
        output_tokens=50,
        model="gpt-4o-mini",
        raw_text="{}",
    )


# ---------------------------------------------------------------------------
# Pure-scorer unit tests
# ---------------------------------------------------------------------------


def test_projection_filed_as_observed_is_flagged():
    records = [_record(metric_value=10000, event_type="case_count",
                       metric_name="confirmed_cases")]
    result = score_extraction(records, TRAP_PROJECTION)
    assert not result.ok
    assert len(result.mishandled) == 1
    assert "filed-as-observed-count" in result.mishandled[0].reasons
    # The (mis)assigned confidence is captured for the calibration metric.
    assert result.trap_confidences == [1.0]


def test_projection_tagged_value_basis_is_not_flagged():
    """Forward-compat: a projection correctly tagged with value_basis must
    NOT trip the observed-count guard, even though it's still a case_count."""
    rec = _record(metric_value=10000, event_type="case_count",
                  metric_name="confirmed_cases")
    rec.value_basis = "projected"  # field the fix will add
    result = score_extraction([rec], TRAP_PROJECTION)
    assert not result.mishandled
    assert result.basis_present == 1
    assert result.basis_correct == 1


def test_projection_omitted_is_not_flagged():
    """Suppressing the projection (no count record for it) is acceptable."""
    rec = _record(metric_value=None, event_type="other",
                  summary="A model projected an outbreak exceeding 10,000 cases.")
    result = score_extraction([rec], TRAP_PROJECTION)
    assert not result.mishandled


def test_control_recall_counts_observed_value():
    from bioscancast.tests.fixtures.insight.extraction_traps import (
        TRAP_CLEAN_CONTROL,
    )
    records = [_record(metric_value=9, event_type="case_count",
                       metric_name="confirmed_cases", iso_country_code="UG")]
    result = score_extraction(records, TRAP_CLEAN_CONTROL)
    assert result.controls_total == 1
    assert result.controls_recalled == 1
    assert result.ok


def test_missing_control_is_not_ok():
    from bioscancast.tests.fixtures.insight.extraction_traps import (
        TRAP_CLEAN_CONTROL,
    )
    result = score_extraction([], TRAP_CLEAN_CONTROL)
    assert result.controls_recalled == 0
    assert not result.ok


def test_offscope_global_tagged_us_is_flagged():
    from bioscancast.tests.fixtures.insight.extraction_traps import (
        TRAP_OFFSCOPE_GLOBAL,
    )
    records = [_record(metric_value=903, event_type="case_count",
                       metric_name="confirmed_cases", iso_country_code="US")]
    result = score_extraction(records, TRAP_OFFSCOPE_GLOBAL)
    assert any("mis-scoped-to-region" in m.reasons for m in result.mishandled)


def test_offscope_global_tagged_global_is_clean():
    from bioscancast.tests.fixtures.insight.extraction_traps import (
        TRAP_OFFSCOPE_GLOBAL,
    )
    # Global cumulative kept global (no US iso) and carrying the qualifier.
    records = [_record(metric_value=903, event_type="case_count",
                       metric_name="confirmed_cases", iso_country_code=None,
                       location="Global", summary="cumulative since 2003 reported to WHO")]
    result = score_extraction(records, TRAP_OFFSCOPE_GLOBAL)
    assert result.ok


def test_weekly_without_qualifier_is_flagged():
    from bioscancast.tests.fixtures.insight.extraction_traps import TRAP_WEEKLY
    records = [
        _record(metric_value=1847, event_type="case_count",
                metric_name="confirmed_cases", iso_country_code="RO"),
        _record(metric_value=56, event_type="case_count",
                metric_name="confirmed_cases", iso_country_code="RO",
                summary="56 cases reported"),  # no week/new qualifier
    ]
    result = score_extraction(records, TRAP_WEEKLY)
    assert any("qualifier-dropped" in m.reasons for m in result.mishandled)


def test_weekly_with_qualifier_is_clean():
    from bioscancast.tests.fixtures.insight.extraction_traps import TRAP_WEEKLY
    records = [
        _record(metric_value=1847, event_type="case_count",
                metric_name="confirmed_cases", iso_country_code="RO"),
        _record(metric_value=56, event_type="case_count",
                metric_name="confirmed_cases", iso_country_code="RO",
                summary="56 new cases in epidemiological week 15"),
    ]
    result = score_extraction(records, TRAP_WEEKLY)
    assert result.ok


def test_subnational_without_state_qualifier_is_flagged():
    from bioscancast.tests.fixtures.insight.extraction_traps import (
        TRAP_SUBNATIONAL,
    )
    records = [
        _record(metric_value=1043, event_type="case_count",
                metric_name="affected_herds", iso_country_code="US"),
        _record(metric_value=312, event_type="case_count",
                metric_name="affected_herds", iso_country_code="US"),  # no Texas
    ]
    result = score_extraction(records, TRAP_SUBNATIONAL)
    assert any("qualifier-dropped" in m.reasons for m in result.mishandled)
    assert result.controls_recalled == 1  # 1043 national still recalled


# ---------------------------------------------------------------------------
# Integration through the real extractor (FakeLLMClient — no network)
# ---------------------------------------------------------------------------


def test_bad_extraction_through_extractor_is_flagged():
    """A model response that files the projection as confirmed_cases should
    be produced by the extractor and caught by the scorer."""
    facts = [
        _fact(  # control: genuine observed US count
            event_type="case_count", metric_name="confirmed_cases",
            metric_value=47, location="United States",
            quote="47 confirmed human cases of H5N1 have been reported in the United States",
        ),
        _fact(  # trap: projection mis-filed as observed
            event_type="case_count", metric_name="confirmed_cases",
            metric_value=10000, location="United States",
            quote="one in 20 simulations projected an outbreak exceeding 10,000 human cases",
        ),
    ]
    client = FakeLLMClient([_response(facts)])
    records, _ = extract_facts_from_chunk(
        TRAP_PROJECTION.chunk, TRAP_PROJECTION.document,
        TRAP_PROJECTION.question, client, model="gpt-4o-mini",
    )
    result = score_extraction(records, TRAP_PROJECTION)
    assert not result.ok
    assert "filed-as-observed-count" in result.mishandled[0].reasons
    assert result.controls_recalled == 1  # the 47 still recalled


def test_good_extraction_through_extractor_passes():
    """A response that keeps the observed count and does NOT file the
    projection as a count passes the scorer."""
    facts = [
        _fact(
            event_type="case_count", metric_name="confirmed_cases",
            metric_value=47, location="United States",
            quote="47 confirmed human cases of H5N1 have been reported in the United States",
        ),
        _fact(  # projection surfaced as context, not a count
            event_type="other", metric_name=None, metric_value=None,
            summary="A modeling study projected an outbreak exceeding 10,000 cases under a worst-case scenario.",
            quote="one in 20 simulations projected an outbreak exceeding 10,000 human cases",
        ),
    ]
    client = FakeLLMClient([_response(facts)])
    records, _ = extract_facts_from_chunk(
        TRAP_PROJECTION.chunk, TRAP_PROJECTION.document,
        TRAP_PROJECTION.question, client, model="gpt-4o-mini",
    )
    result = score_extraction(records, TRAP_PROJECTION)
    assert result.ok


def test_aggregate_smoke():
    """aggregate() produces the headline metrics over a mixed result set."""
    bad = [_record(metric_value=10000, event_type="case_count",
                   metric_name="confirmed_cases")]
    results = [score_extraction(bad, TRAP_PROJECTION)]
    agg = aggregate(results)
    assert agg["traps"] == 1
    assert agg["false_observed_count"] == 1
    assert agg["mean_trap_confidence"] == 1.0


# ---------------------------------------------------------------------------
# Live diagnostic (opt-in: --live + OPENAI_API_KEY)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_extraction_traps_live_diagnostic():
    """Run the real extractor over the trap-set and print a diagnosis report.

    This is a *measurement*, not a pass/fail gate for the current code: it
    quantifies the lurch failure mode (false-observed rate, trap confidence)
    so the fix can be chosen and its effect measured. It only asserts the
    recall guard (a competent extractor must still surface genuine observed
    counts) and that the harness ran.

    Select the model with ``--trap-model`` (default gpt-4o-mini). Run the
    strong model too for the issue #26 cheap-vs-strong comparison.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("needs OPENAI_API_KEY")

    from bioscancast.llm.openai_client import OpenAILLMClient

    model = os.environ.get("TRAP_MODEL", "gpt-4o-mini")
    client = OpenAILLMClient()

    results = []
    print(f"\n=== extraction trap-set diagnostic (model={model}) ===")
    for trap in ALL_TRAPS:
        records, _ = extract_facts_from_chunk(
            trap.chunk, trap.document, trap.question, client, model=model,
        )
        result = score_extraction(records, trap)
        results.append(result)
        flags = ", ".join(
            f"{m.value:g}:{'/'.join(m.reasons)}" for m in result.mishandled
        ) or "clean"
        print(
            f"  {trap.label:<28} mishandled=[{flags}] "
            f"controls={result.controls_recalled}/{result.controls_total}"
        )

    agg = aggregate(results)
    print("  --- aggregate ---")
    for k, v in agg.items():
        print(f"  {k:<24} {v}")

    # Recall guard: genuine observed counts must still be surfaced.
    assert agg["control_recall"] >= 0.5, (
        "Extractor is suppressing genuine observed counts — recall too low."
    )
