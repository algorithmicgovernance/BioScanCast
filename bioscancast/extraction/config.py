from __future__ import annotations

from dataclasses import dataclass


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
