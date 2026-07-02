from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExtractionConfig:
    fetch_timeout_seconds: float = 30.0
    fetch_max_bytes: int = 25_000_000  # 25 MB
    pdf_max_pages: int = 100
    chunk_target_tokens: int = 800
    chunk_max_tokens: int = 1500
    user_agent: str = (
        "BioScanCast/0.1 (+https://github.com/algorithmicgovernance/BioScanCast)"
    )

    # ---- Docling table refiner ----
    enable_docling_refiner: bool = True
    """Toggle the Docling post-pass that refines PDF table sections.

    When False, no Docling imports occur and behaviour is identical to the
    pre-refiner pipeline.
    """

    docling_source_allowlist: List[str] = field(
        default_factory=lambda: [
            "cdc.gov/mmwr/",
            "cdn.who.int/media/docs/default-source/_sage-",
            "cdn.who.int/media/docs/default-source/documents/emergencies/situation-reports/",
        ]
    )
    """Source URL substrings known to contain hard tables. Match triggers Docling unconditionally."""

    docling_sparse_cell_threshold: float = 0.5
    """Non-empty-cell ratio below which a table is flagged as suspect and triggers Docling."""
    impersonate: str = "chrome"
