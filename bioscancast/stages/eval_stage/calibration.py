from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def calibration_table(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    bins: int = 10,
) -> pd.DataFrame:
    """
    Build a simple calibration table from predicted probabilities and outcomes.

    Parameters
    ----------
    probabilities:
        Forecast probabilities, ideally in [0, 1].
    outcomes:
        Observed outcomes encoded as 0/1 values.
    bins:
        Number of probability bins to use.

    Returns
    -------
    pd.DataFrame
        A dataframe with bin-level mean predicted probability, actual frequency,
        and count.
    """
    probs = np.asarray(probabilities, dtype=float)
    obs = np.asarray(outcomes, dtype=float)

    if probs.ndim != 1 or obs.ndim != 1:
        raise ValueError("probabilities and outcomes must be one-dimensional sequences.")

    if len(probs) == 0:
        raise ValueError("probabilities cannot be empty.")

    if len(probs) != len(obs):
        raise ValueError("probabilities and outcomes must have the same length.")

    if np.any(probs < 0):
        raise ValueError("probabilities cannot contain negative values.")

    if np.any((obs != 0) & (obs != 1)):
        raise ValueError("outcomes must be encoded as 0/1 values.")

    # Normalize probabilities if they do not already sum to a valid scale.
    if probs.max() > 1.0:
        probs = probs / 100.0

    probs = np.clip(probs, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "probability": probs,
            "outcome": obs,
        }
    )

    # Use equally spaced bins over [0, 1].
    df["bin"] = pd.cut(
        df["probability"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
    )

    calibration = (
        df.groupby("bin", observed=False)
        .agg(
            mean_probability=("probability", "mean"),
            actual_frequency=("outcome", "mean"),
            count=("outcome", "size"),
        )
        .reset_index()
    )

    return calibration


def plot_calibration_curve(
    calibration_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot a basic calibration curve from a calibration table.

    The output shows mean predicted probability versus observed frequency.
    """
    required = {"mean_probability", "actual_frequency"}
    missing = [col for col in required if col not in calibration_df.columns]
    if missing:
        raise ValueError(
            "calibration_df is missing required columns: " + ", ".join(missing)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plot_df = calibration_df.dropna(subset=["mean_probability", "actual_frequency"]).copy()

    plt.figure(figsize=(6.5, 6.5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(
        plot_df["mean_probability"],
        plot_df["actual_frequency"],
        marker="o",
        label="Model",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()