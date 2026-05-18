from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from bioscancast.stages.eval_stage.compare import compare_sources, compare_sources_by_question_type
from bioscancast.stages.eval_stage.loaders import load_forecasts, load_questions
from bioscancast.stages.eval_stage.scoring import accuracy, log_score, multiclass_brier_score
from bioscancast.stages.eval_stage.visualisation import (
    plot_accuracy_by_source,
    plot_log_score_distribution,
    plot_score_distribution,
    plot_source_comparison,
)


OUTPUT_DIR = Path("outputs")


def _ensure_output_dir() -> None:
    """
    Make sure the output directory exists before writing files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _canonicalize_text(value) -> str:
    """
    Normalize text so bucket matching is more robust.

    This helps with:
    - accidental whitespace
    - en dashes vs hyphens
    - inconsistent casing
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\u2013", "-")  # en dash -> hyphen
    text = text.replace("\u2014", "-")  # em dash -> hyphen
    text = text.strip()

    return text


def _prepare_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the question metadata dataframe.
    """
    df = df.copy()

    required_cols = {"question_id", "question_status", "resolved_option"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "questions dataframe is missing required columns: "
            + ", ".join(missing)
        )

    df["question_id"] = df["question_id"].astype(str).str.strip()
    df["question_status"] = df["question_status"].astype(str).str.lower().str.strip()
    df["resolved_option"] = df["resolved_option"].apply(_canonicalize_text)

    if "question_type" in df.columns:
        df["question_type"] = df["question_type"].astype(str).str.lower().str.strip()
    else:
        df["question_type"] = "unknown"

    if "topic" not in df.columns:
        df["topic"] = ""

    return df


def _prepare_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the forecast dataframe.

    The current CSV uses forecast_version, so we map that to forecast_source
    for downstream comparison and reporting.
    """
    df = df.copy()

    required_cols = {"question_id", "option", "probability"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "forecasts dataframe is missing required columns: "
            + ", ".join(missing)
        )

    df["question_id"] = df["question_id"].astype(str).str.strip()
    df["option"] = df["option"].apply(_canonicalize_text)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    if df["probability"].isna().any():
        bad_rows = df[df["probability"].isna()]
        raise ValueError(
            "Some forecast probabilities could not be parsed as numeric values. "
            f"Problematic rows: {bad_rows.index.tolist()}"
        )

    # Convert percentages to proportions when needed.
    if df["probability"].max() > 1.0:
        df["probability"] = df["probability"] / 100.0

    if "forecast_source" not in df.columns:
        if "forecast_version" in df.columns:
            df["forecast_source"] = df["forecast_version"].astype(str).str.strip()
        else:
            df["forecast_source"] = "forecast"

    df["forecast_source"] = df["forecast_source"].astype(str).str.strip()

    if "forecast_version" in df.columns:
        df["forecast_version"] = df["forecast_version"].astype(str).str.strip()

    return df


def _get_resolved_option_for_group(
    group: pd.DataFrame,
    question_row: pd.Series,
) -> Tuple[str, str]:
    """
    Return the resolved option if it matches one of the forecast options.

    Some rows in the CSV may contain placeholder text such as:
    - 'TBD'
    - 'Ambiguous'
    - 'Resolved on January 1st, 2026'

    Those should not be scored unless they exactly match one of the forecast
    options after normalization.
    """
    resolved_option = _canonicalize_text(question_row["resolved_option"])
    options = [_canonicalize_text(opt) for opt in group["option"].tolist()]

    if resolved_option in options:
        return resolved_option, ""

    lowered = resolved_option.lower()

    if lowered in {"", "tbd", "na", "n/a", "ambiguous"}:
        return "", f"unscorable_status:{lowered or 'empty'}"

    if lowered.startswith("resolved on "):
        return "", "placeholder_resolution_text"

    return "", "resolved_option_not_in_forecast_options"


def build_distribution(group: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Reconstruct one probability distribution from a grouped forecast table.
    """
    group = group.copy()

    if "option_order" in group.columns:
        group = group.sort_values("option_order")

    options = [_canonicalize_text(opt) for opt in group["option"].tolist()]
    probabilities = group["probability"].astype(float).tolist()

    total = sum(probabilities)
    if total <= 0:
        raise ValueError("Forecast probabilities must sum to a positive value.")

    # Normalize to protect against rounding issues.
    probabilities = [p / total for p in probabilities]

    return options, probabilities


def score_all_forecasts(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score each question/source pair and return:
    - a question-level metrics dataframe
    - a skipped-questions dataframe
    """
    results = []
    skipped = []

    grouped = merged_df.groupby(["question_id", "forecast_source"], dropna=False)

    for (question_id, source), group in grouped:
        question_row = group.iloc[0]

        if str(question_row["question_status"]).lower() != "resolved":
            skipped.append(
                {
                    "question_id": question_id,
                    "forecast_source": source,
                    "skip_reason": f"question_status={question_row['question_status']}",
                }
            )
            continue

        options, probabilities = build_distribution(group)
        resolved_option, skip_reason = _get_resolved_option_for_group(
            group=group,
            question_row=question_row,
        )

        if skip_reason:
            skipped.append(
                {
                    "question_id": question_id,
                    "forecast_source": source,
                    "skip_reason": skip_reason,
                }
            )
            continue

        if resolved_option not in options:
            skipped.append(
                {
                    "question_id": question_id,
                    "forecast_source": source,
                    "skip_reason": "resolved_option_missing_after_normalization",
                }
            )
            continue

        true_index = options.index(resolved_option)

        brier = multiclass_brier_score(probabilities, true_index)
        logscore = log_score(probabilities, true_index)
        acc = accuracy(probabilities, true_index)

        results.append(
            {
                "question_id": question_id,
                "topic": question_row.get("topic", ""),
                "question_type": question_row.get("question_type", "unknown"),
                "forecast_source": source,
                "resolved_option": resolved_option,
                "brier_score": brier,
                "log_score": logscore,
                "accuracy": acc,
            }
        )

    results_df = pd.DataFrame(results)
    skipped_df = pd.DataFrame(skipped)

    return results_df, skipped_df


def run_evaluation(
    forecasts_path: str = "bioscancast_forecasts.csv",
    questions_path: str = "bioscancast_questions.csv",
) -> None:
    """
    End-to-end evaluation entry point.

    This reads the CSV files, scores all resolvable forecasts, writes metrics,
    and generates a few basic plots.
    """
    _ensure_output_dir()

    forecasts = load_forecasts(forecasts_path)
    questions = load_questions(questions_path)

    questions = _prepare_questions(questions)
    forecasts = _prepare_forecasts(forecasts)

    resolved_questions = questions[questions["question_status"] == "resolved"].copy()
    ambiguous_questions = questions[questions["question_status"] == "ambiguous"].copy()
    unresolved_questions = questions[questions["question_status"] == "unresolved"].copy()

    if not ambiguous_questions.empty:
        ambiguous_questions.to_csv(OUTPUT_DIR / "ambiguous_questions.csv", index=False)

    if not unresolved_questions.empty:
        unresolved_questions.to_csv(OUTPUT_DIR / "unresolved_questions.csv", index=False)

    merged = forecasts.merge(
        resolved_questions,
        on="question_id",
        how="inner",
        suffixes=("_forecast", "_question"),
    )

    results_df, skipped_df = score_all_forecasts(merged)

    results_path = OUTPUT_DIR / "question_level_metrics.csv"
    skipped_path = OUTPUT_DIR / "skipped_questions.csv"
    summary_path = OUTPUT_DIR / "summary_metrics.csv"
    by_type_path = OUTPUT_DIR / "summary_metrics_by_question_type.csv"

    results_df.to_csv(results_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    if results_df.empty:
        print("No questions could be scored. Check the resolved_option values.")
        return

    summary_df = compare_sources(results_df)
    summary_df.to_csv(summary_path, index=False)

    by_type_df = compare_sources_by_question_type(results_df)
    by_type_df.to_csv(by_type_path, index=False)

    # Main summary plots.
    plot_source_comparison(
        summary_df,
        OUTPUT_DIR / "source_comparison.png",
    )
    plot_accuracy_by_source(
        summary_df,
        OUTPUT_DIR / "accuracy_by_source.png",
    )
    plot_score_distribution(
        results_df,
        OUTPUT_DIR / "brier_distribution.png",
    )
    plot_log_score_distribution(
        results_df,
        OUTPUT_DIR / "log_score_distribution.png",
    )

    # Optional: write a tiny console summary for quick inspection.
    print("\nEvaluation complete.")
    print(summary_df.to_string(index=False))

    if not skipped_df.empty:
        print("\nSkipped questions:")
        print(skipped_df.to_string(index=False))


if __name__ == "__main__":
    run_evaluation()