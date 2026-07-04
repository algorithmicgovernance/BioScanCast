"""Forecasting stage: insight records -> calibrated probability forecast.

Consumes the insight stage's structured facts and a forecast question and
produces a probability distribution over the question's option set, using
superforecaster-style ensemble reasoning. The output conforms to the eval
stage's scoring contract.
"""

from .config import ForecastingConfig
from .pipeline import ForecastingPipeline
from .schemas import (
    ForecastDistribution,
    ForecastRecord,
    ForecastResult,
    SampleForecast,
)

__all__ = [
    "ForecastingConfig",
    "ForecastingPipeline",
    "ForecastResult",
    "SampleForecast",
    "ForecastDistribution",
    "ForecastRecord",
]
