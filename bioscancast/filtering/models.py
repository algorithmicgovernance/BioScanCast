from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ForecastQuestion:
    id: str
    text: str
    created_at: datetime
    target_date: Optional[datetime] = None
    region: Optional[str] = None
    pathogen: Optional[str] = None
    event_type: Optional[str] = None
    resolution_criteria: Optional[str] = None


@dataclass
class SearchResult:
    id: str
    question_id: str
    query_id: str
    engine: str
    url: str
    canonical_url: Optional[str]
    domain: str
    title: str
    snippet: str
    rank: int
    retrieved_at: datetime
    published_date: Optional[datetime] = None
    file_type: Optional[str] = None
    language: Optional[str] = None

    is_official_domain: bool = False
    source_tier: str = "unknown"
    domain_score: float = 0.0
    keyword_overlap_score: float = 0.0
    freshness_score: float = 0.0
    duplicate_cluster_id: Optional[str] = None
    retrieval_reason: Optional[str] = None
    contains_aggregator_forecast: bool = False
    search_stage_score: float = 0.0


@dataclass
class FilterDecision:
    result_id: str
    keep: Optional[bool]
    stage: str
    relevance_score: float
    credibility_score: float
    priority_score: float
    reason_codes: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class FilteredDocument:
    result_id: str
    question_id: str
    url: str
    canonical_url: Optional[str]
    domain: str
    title: str
    snippet: str
    published_date: Optional[datetime]
    file_type: Optional[str]

    relevance_score: float
    credibility_score: float
    final_score: float

    source_tier: str
    is_official_domain: bool
    selection_reasons: List[str]

    extraction_priority: int
    extraction_mode: str
    expected_value: str