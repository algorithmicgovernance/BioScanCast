"""Custom scraper for the WHO cholera upsurge hub (``who_cholera``).

The hub page carries only narrative context, so cholera questions extracted 0
records from it. This resolves the hub to the latest *Multi-country outbreak of
cholera, epidemiological update* PDF at-or-before the cutoff, whose tables carry
the global cumulative case/death totals.
"""

from __future__ import annotations

from datetime import datetime

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.custom_scrapers._who_hub_common import (
    fetch_who_hub_latest_pdf,
)
from bioscancast.stages.extraction.fetcher import FetchResult


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
) -> FetchResult | None:
    return fetch_who_hub_latest_pdf(
        url, "cholera", config=config, as_of_date=as_of_date
    )
