"""Custom scraper for the OWID COVID-19 dashboard (``source_id: ourworldindata_covid``).

Restores COVID parity with the earlier OWID work (issue #38, finding 3): the
``ourworldindata_covid`` source had no scraper module, so extraction fell back
to the generic fetch of a client-side-rendered page and captured zero records
(the exact #34 failure). Fetches OWID's published COVID CSV time series and
surfaces cumulative ``total_cases`` / ``total_deaths`` via the shared OWID core.

Note: the OWID COVID CSV is large (~70 MB). The shared core fetches it with a
raised byte ceiling rather than the generic 25 MB streaming cap.
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

COVID_DATASET = OWIDDataset(
    name="owid-covid",
    csv_url=(
        "https://raw.githubusercontent.com/owid/covid-19-data/master/"
        "public/data/owid-covid-data.csv"
    ),
    label="COVID-19",
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
        COVID_DATASET,
        config=config,
        as_of_date=as_of_date,
        region=region,
        question_text=question_text,
        csv_fetcher=csv_fetcher,
    )
