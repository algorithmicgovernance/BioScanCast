"""Fixtures for end-to-end tests against the 7 real biosecurity documents
already committed under ``data/docling_eval/sources/``.

The integration test runs the real ExtractionPipeline (with the network
fetcher monkey-patched to read on-disk bytes) over each source file and
hands the resulting Documents to the insight pipeline with deterministic
fake LLMs. Re-extracting all 7 sources takes ~5 seconds total with the
in-tree PDF parser, so we don't bother caching.

The module deliberately has no dependency on the local-only smoke script
in ``scripts/eval_insight_on_real_docs.py`` — the test must be
self-contained.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from bioscancast.extraction.config import ExtractionConfig
from bioscancast.extraction.fetcher import FetchResult
from bioscancast.extraction.pipeline import ExtractionPipeline
from bioscancast.filtering.models import FilteredDocument, ForecastQuestion
from bioscancast.llm.base import LLMResponse
from bioscancast.llm.fake_client import FakeLLMClient
from bioscancast.schemas import Document


# Resolve the sources directory relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCES_DIR = _REPO_ROOT / "data" / "docling_eval" / "sources"


SOURCES = [
    {
        "name": "who_mpox_sitrep64",
        "file": "who_mpox_sitrep64.pdf",
        "content_type": "application/pdf",
        "domain": "who.int",
        "url": "https://who.int/publications/m/item/multi-country-outbreak-of-mpox-external-situation-report-64",
        "question": ForecastQuestion(
            id="q-mpox-cases-2026",
            text="How many confirmed mpox cases have been reported globally in 2026?",
            created_at=datetime(2026, 4, 1),
            target_date=datetime(2026, 12, 31),
            region=None,
            pathogen="mpox",
            event_type="case_count",
        ),
    },
    {
        "name": "who_cholera_epi34",
        "file": "who_cholera_epi34.pdf",
        "content_type": "application/pdf",
        "domain": "who.int",
        "url": "https://who.int/publications/m/item/multi-country-outbreak-of-cholera-external-situation-report-34",
        "question": ForecastQuestion(
            id="q-cholera-drc-2026",
            text="How many cholera cases were reported in the Democratic Republic of the Congo?",
            created_at=datetime(2026, 3, 1),
            target_date=datetime(2026, 12, 31),
            region="Democratic Republic of the Congo",
            pathogen="cholera",
            event_type="case_count",
        ),
    },
    {
        "name": "cdc_mmwr_nm_measles",
        "file": "cdc_mmwr_nm_measles.pdf",
        "content_type": "application/pdf",
        "domain": "cdc.gov",
        "url": "https://cdc.gov/mmwr/volumes/74/wr/mm7509a1.htm",
        "question": ForecastQuestion(
            id="q-measles-nm-2025",
            text="How many measles cases were reported in the New Mexico outbreak?",
            created_at=datetime(2026, 3, 15),
            target_date=datetime(2026, 12, 31),
            region="New Mexico",
            pathogen="measles",
            event_type="case_count",
        ),
    },
    {
        "name": "ecdc_cdtr_week16",
        "file": "ecdc_cdtr_week16.pdf",
        "content_type": "application/pdf",
        "domain": "ecdc.europa.eu",
        "url": "https://ecdc.europa.eu/en/publications-data/communicable-disease-threats-report-week-16-2026",
        "question": ForecastQuestion(
            id="q-measles-europe-2026",
            text="How many measles cases have been reported across European countries this year?",
            created_at=datetime(2026, 4, 20),
            target_date=datetime(2026, 12, 31),
            region="Europe",
            pathogen="measles",
            event_type="case_count",
        ),
    },
    {
        "name": "africa_cdc_weekly_apr2026",
        "file": "africa_cdc_weekly_apr2026.pdf",
        "content_type": "application/pdf",
        "domain": "africacdc.org",
        "url": "https://africacdc.org/download/africa-cdc-weekly-event-based-surveillance-april-2026/",
        "question": ForecastQuestion(
            id="q-africa-outbreaks-2026",
            text="What disease outbreaks are currently active across Africa in April 2026?",
            created_at=datetime(2026, 4, 15),
            target_date=datetime(2026, 12, 31),
            region="Africa",
            pathogen=None,
            event_type="outbreak_declared",
        ),
    },
    {
        "name": "cidrap_utah_measles",
        "file": "cidrap_utah_measles.html",
        "content_type": "text/html",
        "domain": "cidrap.umn.edu",
        "url": "https://cidrap.umn.edu/measles/utah-measles-cases-2026",
        "question": ForecastQuestion(
            id="q-measles-utah-2026",
            text="How many measles cases have been confirmed in Utah?",
            created_at=datetime(2026, 4, 25),
            target_date=datetime(2026, 12, 31),
            region="Utah",
            pathogen="measles",
            event_type="case_count",
        ),
    },
    {
        "name": "promed_latest",
        "file": "promed_latest.html",
        "content_type": "text/html",
        "domain": "promedmail.org",
        "url": "https://promedmail.org/promed-posts/",
        "question": ForecastQuestion(
            id="q-promed-h5n1-2026",
            text="What avian influenza H5N1 outbreaks have been reported recently?",
            created_at=datetime(2026, 5, 1),
            target_date=datetime(2026, 12, 31),
            region=None,
            pathogen="H5N1",
            event_type="outbreak_declared",
        ),
    },
]


def make_filtered_doc(source: dict) -> FilteredDocument:
    """Build a minimal FilteredDocument that drives ExtractionPipeline."""
    return FilteredDocument(
        result_id=source["name"],
        question_id=source["question"].id,
        url=source["url"],
        canonical_url=source["url"],
        domain=source["domain"],
        title=source["name"],
        snippet="",
        published_date=None,
        file_type=None,
        relevance_score=0.9,
        credibility_score=0.9,
        final_score=0.9,
        source_tier="official",
        is_official_domain=True,
        selection_reasons=["test"],
        extraction_priority=1,
        extraction_mode="auto",
        expected_value="high",
    )


def _make_fake_fetch(file_path: Path, content_type: str):
    """Return a fetch() replacement that reads on-disk bytes — no network."""
    payload = file_path.read_bytes()

    def fake_fetch(url, *, config=None, as_of_date=None):
        return FetchResult(
            url=url,
            final_url=url,
            status_code=200,
            content_type=content_type,
            content_bytes=payload,
            fetched_at=datetime.now(timezone.utc),
            error=None,
        )

    return fake_fetch


def extract_real_documents() -> dict[str, Document]:
    """Run the real ExtractionPipeline over every source file in SOURCES.

    Returns a mapping from source name to Document. Africa CDC is expected
    to come back with ``status="failed"`` because its PDF is image-only —
    the in-tree parser correctly flags this as ``requires_ocr``.
    """
    config = ExtractionConfig(enable_docling_refiner=False)
    pipeline = ExtractionPipeline(config=config)
    out: dict[str, Document] = {}
    for src in SOURCES:
        path = SOURCES_DIR / src["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        with patch(
            "bioscancast.extraction.pipeline.fetch",
            _make_fake_fetch(path, src["content_type"]),
        ):
            out[src["name"]] = pipeline.extract_one(make_filtered_doc(src))
    return out


def get_source(name: str) -> dict:
    """Look up a source dict by name."""
    for src in SOURCES:
        if src["name"] == name:
            return src
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Fake LLMs for the integration test
# ---------------------------------------------------------------------------


_NUMBER_SENTENCE = re.compile(
    r"[^.\n]*?\b\d[\d,\.]*\b[^.\n]*\.", re.MULTILINE
)


def _extract_chunk_text_from_prompt(user_prompt: str) -> str:
    marker = "CHUNK TEXT:\n"
    idx = user_prompt.find(marker)
    return user_prompt[idx + len(marker):] if idx != -1 else user_prompt


def _pick_quote(chunk_text: str) -> str:
    """Pick a verbatim quote from the chunk — prefer the first sentence
    that contains a number (more interesting for biosecurity facts).
    Fall back to the first non-trivial line."""
    m = _NUMBER_SENTENCE.search(chunk_text)
    if m:
        return m.group(0).strip()[:180]
    for line in chunk_text.splitlines():
        line = line.strip()
        if len(line) > 20:
            return line[:150]
    return chunk_text[:150]


class QuoteEchoingFakeLLM:
    """Reads the chunk text out of the LLM prompt and emits one synthetic
    fact citing a verbatim quote drawn from the chunk. Useful for testing
    the happy path of the hallucination guard on real chunk content
    without any LLM cost."""

    def __init__(self, embedding_client: Optional[FakeLLMClient] = None) -> None:
        self._embed = embedding_client or FakeLLMClient()
        self.calls: list[dict] = []

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        chunk_text = _extract_chunk_text_from_prompt(user)
        quote = _pick_quote(chunk_text)
        fact = {
            "event_type": "case_count",
            "confidence": 0.65,
            "location": None,
            "pathogen": None,
            "metric_name": "events",
            "metric_value": 1.0,
            "metric_unit": "events",
            "event_date": None,
            "summary": "Quote-echoing fake fact for testing.",
            "quote": quote,
        }
        self.calls.append({"quote": quote, "model": model})
        return LLMResponse(
            content={"facts": [fact]},
            input_tokens=100,
            output_tokens=20,
            model=model,
            raw_text=json.dumps({"facts": [fact]}),
        )

    def embed(self, texts, *, model):
        return self._embed.embed(texts, model=model)


class HallucinatingFakeLLM:
    """Always emits a fabricated quote that does not appear in any chunk.
    Every fact should be rejected by the hallucination guard."""

    BOGUS_QUOTE = "THIS QUOTE WAS INVENTED BY THE MODEL AND APPEARS NOWHERE."

    def __init__(self, embedding_client: Optional[FakeLLMClient] = None) -> None:
        self._embed = embedding_client or FakeLLMClient()
        self.calls = 0

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        model: str,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls += 1
        fact = {
            "event_type": "case_count",
            "confidence": 0.9,
            "location": "Atlantis",
            "pathogen": "Imaginarius bogus",
            "metric_name": "confirmed_cases",
            "metric_value": 999.0,
            "metric_unit": "cases",
            "event_date": "2099-01-01",
            "summary": "Fabricated.",
            "quote": self.BOGUS_QUOTE,
        }
        return LLMResponse(
            content={"facts": [fact]},
            input_tokens=100,
            output_tokens=20,
            model=model,
            raw_text=json.dumps({"facts": [fact]}),
        )

    def embed(self, texts, *, model):
        return self._embed.embed(texts, model=model)
