from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.llm.base import LLMResponse
from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.pipeline import ExtractionPipeline
from bioscancast.stages.extraction.custom_scrapers import usda_aphis_livestock
from bioscancast.stages.filtering.models import FilteredDocument, ForecastQuestion
from bioscancast.stages.insight.config import InsightConfig
from bioscancast.stages.insight.pipeline import InsightPipeline


_CSV = """Confirmed Diagnosis,State,County
2024-01-01,IGNORE,NA
2026-01-05,CA,A
2026-01-15,CA,B
2026-02-10,TX,C
2026-03-12,TX,D
2026-03-20,TX,E
2026-04-02,NY,F
2026-04-02,CA,G
2026-04-03,CA,H
"""


def _fetcher(text: str = _CSV):
    return lambda url, config: text


def _html(result) -> str:
    assert result is not None
    assert result.content_type == "text/html"
    return result.content_bytes.decode("utf-8")


def _asof(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def test_renders_requested_analytics_sections():
    result = usda_aphis_livestock.fetch(
        "https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock",
        csv_fetcher=_fetcher(),
    )
    body = _html(result)

    assert "Monthly counts from Confirmed Diagnosis" in body
    assert "Model fit on past 6 months (monthly counts)" in body
    assert "Daily counts from Confirmed Diagnosis" in body
    assert "Model fit on past 30 days (daily counts)" in body
    assert "Per-state summary" in body
    assert "Linear model y = intercept + slope*x" in body
    assert "Exponential model y = a*exp(b*x)" in body
    assert "Cumulative confirmed cases in livestock (from CSV rows): 8" in body
    assert "Affected states and first detected dates" in body
    assert "<td>CA</td><td>2026-01-05</td>" in body
    assert "<td>TX</td><td>2026-02-10</td>" in body
    assert "<td>NY</td><td>2026-04-02</td>" in body


def test_ignores_first_row_before_aggregation():
    result = usda_aphis_livestock.fetch(
        "https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock",
        csv_fetcher=_fetcher(),
    )
    body = _html(result)

    # The first row (2024-01-01, IGNORE) should not appear in output.
    assert "2024-01" not in body
    assert "IGNORE" not in body
    # Remaining months do appear.
    assert "2026-01" in body
    assert "2026-04" in body


def test_as_of_date_applies_cutoff():
    result = usda_aphis_livestock.fetch(
        "https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock",
        as_of_date=_asof("2026-03-31"),
        csv_fetcher=_fetcher(),
    )
    body = _html(result)

    assert "2026-04" not in body
    assert "2026-03" in body


def test_dispatcher_uses_source_id_custom_scraper(monkeypatch):
    from bioscancast.stages.extraction import fetcher as fetcher_mod

    monkeypatch.setattr(usda_aphis_livestock, "_fetch_csv_text", lambda _url, _cfg: _CSV)

    result = fetcher_mod.fetch(
        "https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock",
        config=ExtractionConfig(),
        source_id="usda_aphis_livestock",
    )
    assert result is not None
    assert result.fetch_strategy == "custom:usda_aphis_livestock"
    assert "USDA APHIS HPAI Confirmed Cases in Livestock - analytics snapshot" in _html(result)


def test_returns_none_on_missing_columns():
    bad = "date,state\n2026-01-01,CA\n2026-01-02,TX\n"
    result = usda_aphis_livestock.fetch(
        "https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock",
        csv_fetcher=_fetcher(bad),
    )
    assert result is None


def test_usda_scraper_output_reaches_insight_llm(monkeypatch):
    monkeypatch.setattr(usda_aphis_livestock, "_fetch_csv_text", lambda _url, _cfg: _CSV)

    fdoc = FilteredDocument(
        result_id="r-usda-1",
        question_id="q-usda-1",
        url="https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock",
        canonical_url="https://www.aphis.usda.gov/livestock-poultry-disease/avian/avian-influenza/hpai-detections/hpai-confirmed-cases-livestock",
        domain="aphis.usda.gov",
        title="USDA APHIS HPAI Confirmed Cases in Livestock",
        snippet="Dashboard",
        published_date=None,
        file_type="html",
        relevance_score=1.0,
        credibility_score=1.0,
        final_score=1.0,
        source_tier="official",
        is_official_domain=True,
        selection_reasons=["test"],
        extraction_priority=1,
        extraction_mode="full",
        expected_value="high",
        source_id="usda_aphis_livestock",
        region="United States",
        question_text="How many livestock detections are reported?",
    )

    question = ForecastQuestion(
        id="q-usda-1",
        text="How many HPAI confirmed cases in livestock are reported in the US?",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        region="United States",
        pathogen="H5N1",
        event_type="case_count",
    )

    docs = ExtractionPipeline(config=ExtractionConfig()).run([fdoc])
    assert len(docs) == 1
    assert docs[0].status == "success"
    assert docs[0].chunks

    # Empty fact response is enough to prove the USDA-derived chunks are sent
    # to extraction LLM calls in the insight stage.
    llm = FakeLLMClient([
        LLMResponse(
            content={"facts": []},
            input_tokens=100,
            output_tokens=10,
            model="gpt-4o-mini",
            raw_text='{"facts": []}',
        )
    ])
    insight = InsightPipeline(
        llm_client=llm,
        config=InsightConfig(
            retrieval_top_k=1,
            max_chunks_per_document=1,
            low_survival_top_k=1,
        ),
    ).run(question, docs)

    assert insight.documents_processed == 1
    assert llm.call_count == 1
