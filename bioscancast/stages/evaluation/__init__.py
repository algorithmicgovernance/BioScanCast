"""
BioScanCast evaluation stage.

This package contains:
- CSV loading utilities
- forecast scoring functions
- evaluation pipeline logic
- comparison utilities
- visualisation helpers
"""

from .compare import (
    compare_sources,
    compare_sources_by_question_type,
)

from .loaders import (
    load_forecasts,
    load_questions,
)

from .pipeline import (
    build_distribution,
    run_evaluation,
    score_all_forecasts,
)

from .scoring import (
    accuracy,
    binary_brier_score,
    binary_log_score,
    log_score,
    multiclass_brier_score,
    normalized_entropy,
    ranked_probability_score,
    top_probability,
    true_probability,
)

__all__ = [
    "accuracy",
    "binary_brier_score",
    "binary_log_score",
    "build_distribution",
    "compare_sources",
    "compare_sources_by_question_type",
    "load_forecasts",
    "load_questions",
    "log_score",
    "multiclass_brier_score",
    "normalized_entropy",
    "ranked_probability_score",
    "run_evaluation",
    "score_all_forecasts",
    "top_probability",
    "true_probability",
]
