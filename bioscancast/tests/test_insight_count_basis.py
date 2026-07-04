"""Regression test for the epidemiological count-basis distinction.

This is the deterministic, in-repo version of the count-basis diagnostic:
"282 cumulative", "282 new this week", and "282 active" are the same
metric_name/value but epidemiologically different numbers. Before the
count_basis/time_window fields they flattened into identical structured
records, with the distinction surviving only in free-text summary. These
tests pin the structured distinction so it cannot silently regress.

All tests use FakeLLMClient — no network, no real OpenAI import. The fake
responses stand in for what the (schema-constrained) extractor returns; the
assertions exercise chunk_extractor's propagation of the new fields.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.llm.base import LLMResponse
from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.schemas import Document, DocumentChunk
from bioscancast.stages.filtering.models import ForecastQuestion
from bioscancast.stages.insight.text_extraction.chunk_extractor import extract_facts_from_chunk


_QUESTION = ForecastQuestion(
    id="q-count-basis",
    text="How many confirmed cases of mpox will be reported in Country X?",
    created_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
    region="Country X",
    pathogen="mpox",
)


def _chunk_and_doc(text: str) -> tuple[DocumentChunk, Document]:
    chunk = DocumentChunk(
        chunk_id="c0", chunk_index=0, text=text, chunk_type="prose"
    )
    doc = Document(
        id="d0", result_id="r0", source_url="https://example.org/report",
        domain="example.org", fetched_at=datetime(2026, 7, 3, tzinfo=timezone.utc),
        document_type="html", status="success",
        title="Country X mpox situation report", chunks=[chunk],
    )
    return chunk, doc


def _response(*, count_basis, time_window, surveillance_method, quote,
              data_quality=None) -> LLMResponse:
    return LLMResponse(
        content={
            "facts": [
                {
                    "event_type": "case_count",
                    "confidence": 0.9,
                    "location": "Country X",
                    "pathogen": "mpox",
                    "metric_name": "confirmed_cases",
                    "metric_value": 282,
                    "metric_unit": "cases",
                    "count_basis": count_basis,
                    "time_window": time_window,
                    "surveillance_method": surveillance_method,
                    "data_quality": data_quality,
                    "event_date": "2026-07-03",
                    "summary": None,
                    "quote": quote,
                }
            ]
        },
        input_tokens=200, output_tokens=60, model="gpt-4o-mini", raw_text="{}",
    )


# Same value (282), same metric_name — different epidemiological basis.
_CASES = {
    "cumulative": (
        "As of 3 July 2026, Country X has reported a cumulative total of 282 "
        "confirmed cases of mpox since the start of the outbreak.",
        _response(count_basis="cumulative", time_window="unknown",
                  surveillance_method=None,
                  quote="a cumulative total of 282 confirmed cases of mpox"),
    ),
    "incident": (
        "During the week of 27 June to 3 July 2026, Country X reported 282 new "
        "confirmed cases of mpox, according to laboratory surveillance.",
        _response(count_basis="incident", time_window="week",
                  surveillance_method="laboratory surveillance",
                  quote="Country X reported 282 new confirmed cases of mpox"),
    ),
    "active": (
        "As of 3 July 2026, Country X currently has 282 active confirmed cases "
        "of mpox under isolation.",
        _response(count_basis="active", time_window="unknown",
                  surveillance_method=None,
                  quote="282 active confirmed cases of mpox under isolation"),
    ),
}


def _extract(label: str):
    text, response = _CASES[label]
    chunk, doc = _chunk_and_doc(text)
    client = FakeLLMClient([response])
    records, _ = extract_facts_from_chunk(
        chunk, doc, _QUESTION, client, model="gpt-4o-mini"
    )
    assert len(records) == 1, f"{label}: expected one record"
    return records[0]


def test_count_basis_is_preserved_distinctly():
    """The three snippets share metric_name/value but must differ in
    count_basis — the flattening the meeting flagged."""
    cumulative = _extract("cumulative")
    incident = _extract("incident")
    active = _extract("active")

    # Same surface metric...
    for r in (cumulative, incident, active):
        assert r.metric_name == "confirmed_cases"
        assert r.metric_value == 282.0

    # ...distinct basis.
    assert cumulative.count_basis == "cumulative"
    assert incident.count_basis == "incident"
    assert active.count_basis == "active"
    assert len({cumulative.count_basis, incident.count_basis, active.count_basis}) == 3


def test_time_window_only_for_incident():
    """time_window is meaningful only for incident counts."""
    assert _extract("incident").time_window == "week"
    assert _extract("cumulative").time_window == "unknown"
    assert _extract("active").time_window == "unknown"


def test_surveillance_method_only_when_stated():
    """surveillance_method is captured only where explicitly present."""
    assert _extract("incident").surveillance_method == "laboratory surveillance"
    assert _extract("cumulative").surveillance_method is None
    assert _extract("active").surveillance_method is None


def test_data_quality_propagates_when_present():
    """An explicit data-quality caveat is carried onto the record; a plain
    count leaves it None."""
    text = (
        "Officials cautioned the true number is likely higher than the 282 "
        "confirmed cases, as testing capacity remains limited."
    )
    chunk, doc = _chunk_and_doc(text)
    resp = _response(
        count_basis="cumulative", time_window="unknown",
        surveillance_method=None,
        data_quality="testing capacity limited; true number likely higher",
        quote="the 282 confirmed cases",
    )
    client = FakeLLMClient([resp])
    records, _ = extract_facts_from_chunk(
        chunk, doc, _QUESTION, client, model="gpt-4o-mini"
    )
    assert records[0].data_quality == "testing capacity limited; true number likely higher"
    # Clean count -> no caveat.
    assert _extract("cumulative").data_quality is None


def test_fields_default_to_none_when_absent():
    """Backward compatibility: a response omitting the new keys (e.g. an
    old fixture) must not raise — the fields fall back to None."""
    text = (
        "On 3 July 2026, Country X's Ministry of Health reported 282 confirmed "
        "cases of mpox."
    )
    chunk, doc = _chunk_and_doc(text)
    legacy = LLMResponse(
        content={
            "facts": [
                {
                    "event_type": "case_count",
                    "confidence": 0.8,
                    "location": "Country X",
                    "pathogen": "mpox",
                    "metric_name": "confirmed_cases",
                    "metric_value": 282,
                    "metric_unit": "cases",
                    "event_date": "2026-07-03",
                    "summary": None,
                    "quote": "reported 282 confirmed cases of mpox",
                }
            ]
        },
        input_tokens=100, output_tokens=20, model="gpt-4o-mini", raw_text="{}",
    )
    client = FakeLLMClient([legacy])
    records, _ = extract_facts_from_chunk(
        chunk, doc, _QUESTION, client, model="gpt-4o-mini"
    )
    assert len(records) == 1
    assert records[0].count_basis is None
    assert records[0].time_window is None
    assert records[0].surveillance_method is None
    assert records[0].data_quality is None
