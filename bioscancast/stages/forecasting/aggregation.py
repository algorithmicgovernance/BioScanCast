"""Ensemble aggregation for the forecasting stage.

Pure functions: given several probability vectors (the ensemble members),
pool them into one. All functions return a normalized vector.

The default pool is the *log opinion pool* (normalized geometric mean of
probabilities). For a binary question this is exactly the geometric mean
of odds — the pooling rule superforecasting research favours because it
resists a single extreme member dragging the consensus, while staying
sharper than the arithmetic mean.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Clip away from 0/1 before taking logs so a confident-but-wrong member
# can't blow the geometric mean up to +/- infinity.
_EPSILON = 1e-9


def _stack(samples: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and stack ensemble members into an (n_samples, n_options)
    array of normalized rows."""
    if not samples:
        raise ValueError("Cannot aggregate an empty ensemble.")
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Ensemble members must all be vectors of equal length.")
    if np.any(arr < 0):
        raise ValueError("Probabilities cannot be negative.")
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each ensemble member must sum to a positive value.")
    return arr / row_sums


def _normalize(vec: np.ndarray) -> List[float]:
    total = float(vec.sum())
    if total <= 0:
        # Degenerate; fall back to uniform rather than raising.
        n = len(vec)
        return [1.0 / n] * n
    return (vec / total).tolist()


def geometric_mean_of_odds(samples: Sequence[Sequence[float]]) -> List[float]:
    """Log opinion pool: normalized geometric mean of the member vectors.

    Reduces to the geometric mean of odds for the binary (two-option)
    case.
    """
    arr = _stack(samples)
    arr = np.clip(arr, _EPSILON, 1.0)
    log_mean = np.mean(np.log(arr), axis=0)
    pooled = np.exp(log_mean)
    return _normalize(pooled)


def arithmetic_mean(samples: Sequence[Sequence[float]]) -> List[float]:
    """Linear opinion pool: elementwise mean of the member vectors."""
    arr = _stack(samples)
    return _normalize(arr.mean(axis=0))


def median(samples: Sequence[Sequence[float]]) -> List[float]:
    """Elementwise median, then renormalize (the per-option medians need
    not sum to 1)."""
    arr = _stack(samples)
    return _normalize(np.median(arr, axis=0))


def extremize(probabilities: Sequence[float], strength: float) -> List[float]:
    """Sharpen a distribution away from uniform.

    Raises each probability to ``1 + strength`` and renormalizes.
    ``strength == 0`` is a no-op (exponent 1). Negative strengths flatten
    toward uniform.
    """
    arr = np.asarray(probabilities, dtype=float)
    if strength == 0.0:
        return _normalize(arr)
    arr = np.clip(arr, _EPSILON, 1.0)
    return _normalize(np.power(arr, 1.0 + strength))


_METHODS = {
    "geometric_mean_of_odds": geometric_mean_of_odds,
    "mean": arithmetic_mean,
    "median": median,
}


def aggregate(
    samples: Sequence[Sequence[float]],
    method: str = "geometric_mean_of_odds",
    *,
    extremize_strength: float = 0.0,
    extremize_gate: float | None = None,
) -> List[float]:
    """Pool ensemble members with ``method``, then optionally extremize.

    ``extremize_gate``, when set, only extremizes if the pooled
    distribution's peak probability is at or above the threshold. This
    concentrates sharpening on already-decisive forecasts and leaves
    diffuse (and so more often wrong) ones untouched — offline analysis
    on the historical-replay set found gating robustly beats ungated or
    static extremizing (see ``scripts/eval_forecast_calibration.py``).
    ``None`` (the default) extremizes unconditionally.

    Returns a normalized probability vector.
    """
    try:
        fn = _METHODS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown aggregation method {method!r}. "
            f"Choose from {sorted(_METHODS)}."
        ) from exc
    pooled = fn(samples)
    if extremize_strength and (extremize_gate is None or max(pooled) >= extremize_gate):
        pooled = extremize(pooled, extremize_strength)
    return pooled
