"""Custom scraper for the WHO "Influenza at the human-animal interface" monthly
risk-assessment hub (``who_h5_hai``).

The monthly-risk-assessment-summary page is an index: it lists dated
*Influenza at the human-animal interface summary and assessment* item pages
(``who.int/publications/m/item/influenza-at-the-human-animal-interface-...``),
each of which links a ``cdn.who.int`` PDF whose tables/prose carry the cumulative
human case counts by A(H5) subtype. The index page itself is context-only, so H5
human-case questions (q6 case count, q8 predominant subtype) extracted 0 records
from it.

This resolves the hub to the latest assessment at-or-before the cutoff and returns
that item's PDF, exactly as ``who_cholera`` does for the cholera hub, so the
existing PDF parser extracts the numbers. Falls back (return ``None``) to the
generic fetch of the index page if resolution fails.
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
    # "human-animal" selects the HAI summary items (their slug/text both carry
    # "human-animal interface") out of the ~150 mixed publication links on the
    # monthly-risk-assessment index, and excludes the FAO/WOAH joint-assessment
    # and vaccine-composition items that don't hold the human-case counts.
    return fetch_who_hub_latest_pdf(
        url, "human-animal", config=config, as_of_date=as_of_date
    )
