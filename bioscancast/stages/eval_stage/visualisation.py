from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent_dir(output_path: str | Path) -> Path:
    """
    Make sure the destination folder exists before saving a figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_source_comparison(
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot mean Brier, log score, and accuracy by forecast source.

    Expected columns:
    - forecast_source
    - mean_brier_score
    - mean_log_score
    - mean_accuracy
    """
    required = {
        "forecast_source",
        "mean_brier_score",
        "mean_log_score",
        "mean_accuracy",
    }
    missing = [col for col in required if col not in summary_df.columns]
    if missing:
        raise ValueError(
            "summary_df is missing required columns: "
            + ", ".join(missing)
        )

    output_path = _ensure_parent_dir(output_path)

    plot_df = summary_df.set_index("forecast_source")[
        ["mean_brier_score", "mean_log_score", "mean_accuracy"]
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(9, 5),
        rot=0,
    )

    ax.set_xlabel("Forecast source")
    ax.set_ylabel("Metric value")
    ax.set_title("Forecast performance by source")
    ax.legend(title="Metric")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_distribution(
    results_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot the distribution of Brier scores across questions.
    """
    if "brier_score" not in results_df.columns:
        raise ValueError("results_df must contain a 'brier_score' column.")

    output_path = _ensure_parent_dir(output_path)

    plt.figure(figsize=(8, 5))
    plt.hist(results_df["brier_score"].dropna(), bins=10)
    plt.xlabel("Brier score")
    plt.ylabel("Count")
    plt.title("Distribution of question-level Brier scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_log_score_distribution(
    results_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot the distribution of log scores across questions.
    """
    if "log_score" not in results_df.columns:
        raise ValueError("results_df must contain a 'log_score' column.")

    output_path = _ensure_parent_dir(output_path)

    plt.figure(figsize=(8, 5))
    plt.hist(results_df["log_score"].dropna(), bins=10)
    plt.xlabel("Log score")
    plt.ylabel("Count")
    plt.title("Distribution of question-level log scores")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_accuracy_by_source(
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot mean categorical accuracy by forecast source.
    """
    required = {"forecast_source", "mean_accuracy"}
    missing = [col for col in required if col not in summary_df.columns]
    if missing:
        raise ValueError(
            "summary_df is missing required columns: "
            + ", ".join(missing)
        )

    output_path = _ensure_parent_dir(output_path)

    plot_df = summary_df.set_index("forecast_source")[["mean_accuracy"]]

    ax = plot_df.plot(kind="bar", figsize=(7, 5), legend=False, rot=0)
    ax.set_xlabel("Forecast source")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean accuracy by source")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_probability_distribution(
    options: Sequence[str],
    probabilities: Sequence[float],
    title: str,
    output_path: str | Path,
) -> None:
    """
    Plot a single forecast distribution as a bar chart.
    """
    if len(options) != len(probabilities):
        raise ValueError("options and probabilities must have the same length.")

    output_path = _ensure_parent_dir(output_path)

    plt.figure(figsize=(9, 5))
    plt.bar(list(options), list(probabilities))
    plt.xlabel("Outcome")
    plt.ylabel("Probability")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_question_level_comparison(
    comparison_df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
) -> None:
    """
    Plot a question-level metric comparison table.

    Useful when the dataframe contains one row per question and columns like:
    - brier_score_human
    - brier_score_llm
    """
    if "question_id" not in comparison_df.columns:
        raise ValueError("comparison_df must contain a 'question_id' column.")

    metric_columns = [
        col for col in comparison_df.columns if col.startswith(f"{metric}_")
    ]

    if len(metric_columns) < 2:
        raise ValueError(
            f"comparison_df must contain at least two columns starting with '{metric}_'."
        )

    output_path = _ensure_parent_dir(output_path)

    plot_df = comparison_df.set_index("question_id")[metric_columns]

    ax = plot_df.plot(kind="bar", figsize=(11, 5))
    ax.set_xlabel("Question ID")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Question-level {metric.replace('_', ' ')} comparison")
    ax.legend(title="Source")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()