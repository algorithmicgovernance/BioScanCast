from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def _read_csv(path: PathLike) -> pd.DataFrame:
    """
    Read one of the BioScanCast CSV files with the correct separator,
    encoding, and decimal format.
    """
    return pd.read_csv(
        path,
        sep=";",
        encoding="cp1252",
        decimal=",",
    )


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common text fields so matching is stable across files.

    This mainly helps with:
    - spacing
    - dash variants
    - accidental surrounding whitespace
    """
    text_columns = df.select_dtypes(include="object").columns

    for col in text_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\u2013", "-", regex=False)  # en dash -> hyphen
            .str.replace("\u2014", "-", regex=False)  # em dash -> hyphen
            .str.strip()
        )

    return df


def load_questions(path: PathLike) -> pd.DataFrame:
    """
    Load the question metadata CSV.

    Expected columns:
    - question_id
    - topic
    - question_text
    - question_type
    - resolution_criteria
    - created_date
    - question_status
    - resolved_option
    - comparison_to_outcome
    - takeaways
    - relevant_links
    """
    df = _read_csv(path)
    df = _clean_text_columns(df)

    if "created_date" in df.columns:
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")

    if "question_status" in df.columns:
        df["question_status"] = df["question_status"].str.lower()

    return df


def load_forecasts(path: PathLike) -> pd.DataFrame:
    """
    Load the forecasts CSV.

    Expected columns:
    - question_id
    - forecast_version
    - option
    - probability
    """
    df = _read_csv(path)
    df = _clean_text_columns(df)

    if "probability" not in df.columns:
        raise ValueError("Forecast file must contain a 'probability' column.")

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")

    if df["probability"].isna().any():
        bad_rows = df[df["probability"].isna()]
        raise ValueError(
            "Some forecast probabilities could not be parsed as numbers. "
            f"Problematic rows: {bad_rows.index.tolist()}"
        )

    if "forecast_version" in df.columns:
        df["forecast_version"] = df["forecast_version"].str.strip()

    if "question_id" in df.columns:
        df["question_id"] = df["question_id"].str.strip()

    return df