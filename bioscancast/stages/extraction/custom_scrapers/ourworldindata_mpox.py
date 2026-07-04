"""Custom scraper for the OWID mpox dashboard (``source_id: ourworldindata_mpox``).

Fetches OWID's published mpox CSV time series (canonical GitHub raw — the
grapher/explorer ``.csv`` slug endpoints from #34 are unstable and the explorer
endpoint only exposes 7-day-smoothed *new* cases, not the cumulative totals the
benchmark questions resolve on; see #38) and surfaces cumulative ``total_cases``
/ ``total_deaths`` via the shared OWID core.
"""

from __future__ import annotations

from datetime import datetime

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.custom_scrapers._owid_common import (
    CsvFetcher,
    OWIDDataset,
    fetch_owid,
)
from bioscancast.stages.extraction.fetcher import FetchResult

MPOX_DATASET = OWIDDataset(
    name="owid-mpox",
    csv_url=(
        "https://raw.githubusercontent.com/owid/monkeypox/main/"
        "owid-monkeypox-data.csv"
    ),
    label="mpox",
)


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
    csv_fetcher: CsvFetcher | None = None,
) -> FetchResult | None:
    return fetch_owid(
        url,
        MPOX_DATASET,
        config=config,
        as_of_date=as_of_date,
        region=region,
        question_text=question_text,
        csv_fetcher=csv_fetcher,
    )
