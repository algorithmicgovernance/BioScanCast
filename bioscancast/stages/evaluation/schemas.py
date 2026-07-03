from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ForecastRecord:
    """
    In-memory representation of one forecast row.

    This is useful when you want to convert the CSV rows into a cleaner,
    typed object before scoring or plotting.
    """
    question_id: str
    forecast_source: str
    option: str
    probability: float
    forecast_version: Optional[str] = None


@dataclass(frozen=True)
class QuestionRecord:
    """
    In-memory representation of one question row from the questions CSV.
    """
    question_id: str
    topic: str
    question_text: str
    question_type: str
    resolution_criteria: str
    created_date: Optional[str]
    question_status: str
    resolved_option: str
    comparison_to_outcome: str
    takeaways: str
    relevant_links: str


@dataclass(frozen=True)
class ScoredQuestion:
    """
    A scored forecast result for one question and one source.
    """
    question_id: str
    topic: str
    question_type: str
    forecast_source: str
    resolved_option: str
    brier_score: float
    log_score: float
    accuracy: int


@dataclass(frozen=True)
class ForecastDistribution:
    """
    A normalized probability distribution over discrete outcomes.
    """
    question_id: str
    forecast_source: str
    probabilities: Dict[str, float]