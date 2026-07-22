from __future__ import annotations

from datetime import datetime, timezone

from bioscancast.llm.base import LLMResponse
from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.custom_scrapers import paho_oropouche_portal
from bioscancast.stages.extraction.pipeline import ExtractionPipeline
from bioscancast.stages.filtering.models import FilteredDocument, ForecastQuestion
from bioscancast.stages.insight.config import InsightConfig
from bioscancast.stages.insight.pipeline import InsightPipeline


_CSV = """Año,Semanas Epi,Week Lab Confirmed,Pais
2026,1,10,Brazil
2026,1,5,Peru
2026,2,7,Brazil
2026,3,0,Bolivia
2026,20,9,Brazil
2026,21,4,Peru
2026,22,3,Colombia
2026,23,8,Brazil
2026,24,2,Peru
"""


def _fetcher(text: str = _CSV):
    return lambda url, config: text


def _html(result) -> str:
    assert result is not None
    assert result.content_type == "text/html"
    return result.content_bytes.decode("utf-8")


def _asof(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def test_renders_weekly_and_country_analytics():
    result = paho_oropouche_portal.fetch(
        "https://www.paho.org/en/arbo-portal/arbo-portal-oropouche",
        csv_fetcher=_fetcher(),
    )
    body = _html(result)

    assert "Weekly counts from Año + Semanas Epi" in body
    assert "Counts by year_week" in body
    assert "2026-W01" in body
    assert "Cumulative confirmed count across all weeks" in body
    assert "Model fit on past 24 weeks" in body
    assert "Country/state grouping from year_week + Pais" in body
    assert "Per-country summary" in body
    assert "Affected states/countries (Pais values):" in body


def test_as_of_date_filters_future_weeks():
    result = paho_oropouche_portal.fetch(
        "https://www.paho.org/en/arbo-portal/arbo-portal-oropouche",
        as_of_date=_asof("2026-05-25"),
        csv_fetcher=_fetcher(),
    )
    body = _html(result)

    assert "2026-W23" not in body
    assert "2026-W24" not in body


def test_dispatcher_uses_source_id_custom_scraper(monkeypatch):
    from bioscancast.stages.extraction import fetcher as fetcher_mod

    monkeypatch.setattr(paho_oropouche_portal, "_fetch_full_data_csv", lambda _url, _cfg: _CSV)

    result = fetcher_mod.fetch(
        "https://www.paho.org/en/arbo-portal/arbo-portal-oropouche",
        config=ExtractionConfig(),
        source_id="paho_oropouche_portal",
    )
    assert result is not None
    assert result.fetch_strategy == "custom:paho_oropouche_portal"
    assert "PAHO Oropouche weekly analytics snapshot" in _html(result)


def test_returns_none_on_missing_required_columns():
    bad = "year,week,cases,country\n2026,1,10,Brazil\n"
    result = paho_oropouche_portal.fetch(
        "https://www.paho.org/en/arbo-portal/arbo-portal-oropouche",
        csv_fetcher=_fetcher(bad),
    )
    assert result is None


def test_oropouche_scraper_output_reaches_insight_llm(monkeypatch):
    monkeypatch.setattr(paho_oropouche_portal, "_fetch_full_data_csv", lambda _url, _cfg: _CSV)

    fdoc = FilteredDocument(
        result_id="r-orov-1",
        question_id="q-orov-1",
        url="https://www.paho.org/en/arbo-portal/arbo-portal-oropouche",
        canonical_url="https://www.paho.org/en/arbo-portal/arbo-portal-oropouche",
        domain="www.paho.org",
        title="PAHO Oropouche portal",
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
        source_id="paho_oropouche_portal",
        region="Americas",
        question_text="How many countries/territories report Oropouche?",
    )

    question = ForecastQuestion(
        id="q-orov-1",
        text="How many Americas countries or territories report confirmed Oropouche?",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        region="Americas",
        pathogen="Oropouche",
        event_type="case_count",
    )

    docs = ExtractionPipeline(config=ExtractionConfig()).run([fdoc])
    assert len(docs) == 1
    assert docs[0].status == "success"
    assert docs[0].chunks

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


# The live "Full Data" vudcsv export carries extra columns (SE, Iso3166,
# Actualizado, Dominio, ...) and a different column order than the minimal
# fixture above; the parser must resolve the four it needs and ignore the rest.
_FULL_DATA_CSV = (
    "Week Lab Confirmed,Semanas Epi,Pais,Año,SE,Iso3166,Actualizado,Dominio,Fuente\n"
    "1,2,Brasil,2026,15,76,2026-04-28,PAHO internal use only,Boletim\n"
    "3,5,Brasil,2026,15,76,2026-04-28,PAHO internal use only,Boletim\n"
    "2,4,Panamá,2026,15,591,2026-04-28,PAHO internal use only,MINSA\n"
)


def test_parses_full_data_column_superset():
    result = paho_oropouche_portal.fetch(
        "https://www.paho.org/en/arbo-portal/arbo-portal-oropouche",
        csv_fetcher=_fetcher(_FULL_DATA_CSV),
    )
    body = _html(result)
    assert "Weekly counts from Año + Semanas Epi" in body
    assert "Per-country summary" in body
    # Both reporting countries surface in the per-country grouping.
    assert "Brasil" in body
    assert "Panamá" in body
