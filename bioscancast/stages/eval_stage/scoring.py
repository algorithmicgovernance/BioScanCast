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