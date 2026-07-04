from __future__ import annotations

from typing import Sequence

import numpy as np

EPSILON = 1e-15


def _to_probability_vector(probabilities: Sequence[float]) -> np.ndarray:
    """
    Convert a sequence of raw values into a valid probability vector.

    This function is intentionally forgiving:
    - It accepts values already in [0, 1]
    - It also accepts percentage-like values that sum to ~100
    - It normalizes the vector so the final sum is exactly 1.0
    """
    probs = np.asarray(probabilities, dtype=float)

    if probs.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional sequence.")

    if len(probs) == 0:
        raise ValueError("Probabilities cannot be empty.")

    if np.any(probs < 0):
        raise ValueError("Probabilities cannot contain negative values.")

    total = probs.sum()

    if total <= 0:
        raise ValueError("Probabilities must sum to a positive value.")

    probs = probs / total

    return probs


def multiclass_brier_score(
    probabilities: Sequence[float],
    true_index: int,
) -> float:
    """
    Compute the multiclass Brier score.

    Lower is better. A perfect forecast scores 0.0.
    """
    probs = _to_probability_vector(probabilities)

    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")

    outcome = np.zeros(len(probs), dtype=float)
    outcome[true_index] = 1.0

    return float(np.sum((probs - outcome) ** 2))


def binary_brier_score(
    probability_yes: float,
    outcome_yes: int,
) -> float:
    """
    Compute the Brier score for a binary YES/NO forecast.

    `probability_yes` should be the forecast probability assigned to YES.
    `outcome_yes` should be 1 if the event happened, otherwise 0.
    """
    p_yes = float(probability_yes)

    if p_yes < 0:
        raise ValueError("Probability cannot be negative.")

    if p_yes > 1:
        p_yes = p_yes / 100.0

    p_yes = min(max(p_yes, 0.0), 1.0)

    if outcome_yes not in (0, 1):
        raise ValueError("outcome_yes must be either 0 or 1.")

    return float((p_yes - outcome_yes) ** 2)


def log_score(
    probabilities: Sequence[float],
    true_index: int,
) -> float:
    """
    Compute the logarithmic score for a multiclass forecast.

    Lower is better. A perfect forecast approaches 0.0.
    """
    probs = _to_probability_vector(probabilities)

    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")

    p_true = float(probs[true_index])
    p_true = np.clip(p_true, EPSILON, 1.0 - EPSILON)

    return float(-np.log(p_true))


def binary_log_score(probability_yes: float, outcome_yes: int) -> float:
    """
    Compute the log score for a binary YES/NO forecast.
    """
    p_yes = float(probability_yes)

    if p_yes < 0:
        raise ValueError("Probability cannot be negative.")

    if p_yes > 1:
        p_yes = p_yes / 100.0

    p_yes = min(max(p_yes, 0.0), 1.0)
    p_no = 1.0 - p_yes

    if outcome_yes not in (0, 1):
        raise ValueError("outcome_yes must be either 0 or 1.")

    p_true = p_yes if outcome_yes == 1 else p_no
    p_true = np.clip(p_true, EPSILON, 1.0 - EPSILON)

    return float(-np.log(p_true))


def accuracy(
    probabilities: Sequence[float],
    true_index: int,
) -> int:
    """
    Return 1 if the most likely bucket matches the resolved bucket.
    """
    probs = _to_probability_vector(probabilities)

    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")

    predicted_index = int(np.argmax(probs))

    return int(predicted_index == true_index)


def true_probability(
    probabilities: Sequence[float],
    true_index: int,
) -> float:
    """
    Return the probability mass the forecast assigned to the resolved bucket.

    Higher is better. A perfect forecast approaches 1.0.
    """
    probs = _to_probability_vector(probabilities)

    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")

    return float(probs[true_index])


def top_probability(probabilities: Sequence[float]) -> float:
    """
    Return the probability mass on the single most likely bucket.

    This is a simple confidence proxy: values near 1.0 mean the forecast
    concentrated on one bucket, values near 1/n mean it spread mass evenly.
    """
    probs = _to_probability_vector(probabilities)

    return float(np.max(probs))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """
    Return the Shannon entropy of the forecast, normalized to [0, 1].

    0.0 means all mass sits on one bucket (maximally confident); 1.0 means
    mass is spread uniformly across every bucket (maximally uncertain).
    """
    probs = _to_probability_vector(probabilities)

    n = len(probs)
    if n <= 1:
        return 0.0

    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))

    return entropy / float(np.log(n))


def ranked_probability_score(
    probabilities: Sequence[float],
    true_index: int,
) -> float:
    """
    Compute the Ranked Probability Score for an ordinal multiclass forecast.

    RPS compares the cumulative forecast distribution against the cumulative
    outcome (a step function that jumps to 1 at the resolved bucket), summing
    the squared differences. It is normalized by (n - 1) so the result lies in
    [0, 1]; lower is better.

    IMPORTANT: RPS is only meaningful when the buckets have a natural order,
    because it rewards putting mass *near* the true bucket. The order used is
    the order of ``probabilities`` as passed in, so callers are responsible for
    ordering the buckets (``build_distribution`` sorts by ``option_order`` when
    that column is present). For unordered/nominal questions (e.g. yes/no)
    the ordering is arbitrary and RPS should be interpreted with care.
    """
    probs = _to_probability_vector(probabilities)

    n = len(probs)
    if true_index < 0 or true_index >= n:
        raise IndexError("true_index is out of bounds for the probability vector.")

    if n <= 1:
        return 0.0

    outcome = np.zeros(n, dtype=float)
    outcome[true_index] = 1.0

    cum_pred = np.cumsum(probs)
    cum_outcome = np.cumsum(outcome)

    return float(np.sum((cum_pred - cum_outcome) ** 2) / (n - 1))
