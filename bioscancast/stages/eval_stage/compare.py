from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


METRIC_COLUMNS = [
    'brier_score',
    'log_score',
    'accuracy',
    'accuracy_error',
    'rps',
    'top_probability',
    'normalized_entropy',
    'true_probability',
]


def _version_sort_key(value):
    text = str(value)
    try:
        return (0, float(text))
    except Exception:
        return (1, text)


def _ordered_unique(values: Sequence[object]) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
    return seen


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError('results_df is missing required columns: ' + ', '.join(missing))


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').dropna().astype(float)


def _summarise_group(group: pd.DataFrame, key_columns: Sequence[str]) -> dict:
    row: dict = {col: group.iloc[0][col] for col in key_columns}
    row['n_questions'] = int(group['question_id'].nunique()) if 'question_id' in group.columns else int(len(group))
    if 'forecast_version' in group.columns:
        row['n_versions'] = int(group['forecast_version'].nunique())

    for metric in METRIC_COLUMNS:
        if metric not in group.columns:
            continue
        values = _clean_numeric(group[metric])
        if values.empty:
            row[f'mean_{metric}'] = float('nan')
            row[f'median_{metric}'] = float('nan')
            row[f'std_{metric}'] = float('nan')
            row[f'q1_{metric}'] = float('nan')
            row[f'q3_{metric}'] = float('nan')
            continue
        row[f'mean_{metric}'] = float(values.mean())
        row[f'median_{metric}'] = float(values.median())
        row[f'std_{metric}'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row[f'q1_{metric}'] = float(values.quantile(0.25))
        row[f'q3_{metric}'] = float(values.quantile(0.75))
    return row


def _summarise_by_group(results_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    _require_columns(results_df, {'question_id', *group_cols})
    rows = [_summarise_group(group, group_cols) for _, group in results_df.groupby(list(group_cols), dropna=False)]
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    sort_cols = [col for col in group_cols if col in summary.columns]
    if 'forecast_version' in summary.columns:
        summary = summary.sort_values(
            ['forecast_version'] + [c for c in sort_cols if c != 'forecast_version'],
            key=lambda s: s.map(_version_sort_key) if s.name == 'forecast_version' else s.astype(str),
        )
    else:
        summary = summary.sort_values(sort_cols)
    return summary.reset_index(drop=True)


def compare_sources(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scoring metrics by forecast source."""
    _require_columns(results_df, {'question_id', 'forecast_source'})
    summary = _summarise_by_group(results_df, ['forecast_source'])
    if 'median_brier_score' in summary.columns:
        summary = summary.sort_values(['median_brier_score', 'forecast_source'], ascending=[True, True]).reset_index(drop=True)
    return summary


def compare_sources_over_time(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scoring metrics by forecast version and source."""
    _require_columns(results_df, {'question_id', 'forecast_source', 'forecast_version'})
    summary = _summarise_by_group(results_df, ['forecast_version', 'forecast_source'])
    if not summary.empty:
        summary = summary.sort_values(
            ['forecast_version', 'forecast_source'],
            key=lambda s: s.map(_version_sort_key) if s.name == 'forecast_version' else s.astype(str),
        ).reset_index(drop=True)
    return summary


def compare_sources_by_question_type(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scoring metrics by forecast source and question type."""
    _require_columns(results_df, {'question_id', 'forecast_source'})
    if 'question_type' not in results_df.columns:
        raise ValueError("results_df must contain a 'question_type' column to compare by question type.")

    summary = _summarise_by_group(results_df, ['forecast_source', 'question_type'])
    if not summary.empty:
        summary = summary.sort_values(['forecast_source', 'question_type']).reset_index(drop=True)
    return summary


def rank_sources_over_time(summary_df: pd.DataFrame, *, metric_column: str = 'median_brier_score', ascending: bool = True) -> pd.DataFrame:
    """Rank sources within each version using the selected metric."""
    required = {'forecast_version', 'forecast_source', metric_column}
    _require_columns(summary_df, required)

    rows = []
    for version, group in summary_df.groupby('forecast_version', dropna=False):
        ordered = group.sort_values([metric_column, 'forecast_source'], ascending=[ascending, True]).reset_index(drop=True)
        ranks = ordered[metric_column].rank(method='dense', ascending=ascending).astype(int)
        for idx, (_, row) in enumerate(ordered.iterrows()):
            rows.append(
                {
                    'forecast_version': version,
                    'forecast_source': row['forecast_source'],
                    'metric_column': metric_column,
                    'metric_value': float(row[metric_column]),
                    'rank': int(ranks.iloc[idx]),
                }
            )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(
        ['forecast_version', 'rank', 'forecast_source'],
        key=lambda s: s.map(_version_sort_key) if s.name == 'forecast_version' else s.astype(str),
    )
    return ranked.reset_index(drop=True)


def relative_improvement_over_time(results_df: pd.DataFrame, baseline_version: str | None = None) -> pd.DataFrame:
    """Measure change versus the baseline forecast version for each source and metric.

    Positive values mean the forecast improved relative to the baseline because
    the evaluated metrics are lower-is-better.
    """
    required = {'question_id', 'forecast_source', 'forecast_version'} | set(METRIC_COLUMNS)
    _require_columns(results_df, required)

    versions = sorted(_ordered_unique(results_df['forecast_version'].tolist()), key=_version_sort_key)
    if not versions:
        return pd.DataFrame()
    if baseline_version is None:
        baseline_version = versions[0]

    rows = []
    for source, source_df in results_df.groupby('forecast_source', dropna=False):
        baseline_df = source_df[source_df['forecast_version'].astype(str) == str(baseline_version)]
        if baseline_df.empty:
            continue
        for version in versions:
            current_df = source_df[source_df['forecast_version'].astype(str) == str(version)]
            merged = baseline_df[['question_id', *METRIC_COLUMNS]].merge(
                current_df[['question_id', *METRIC_COLUMNS]],
                on='question_id',
                suffixes=('_baseline', '_current'),
            )
            if merged.empty:
                continue
            for metric in METRIC_COLUMNS:
                pair = merged[[f'{metric}_baseline', f'{metric}_current']].dropna()
                if pair.empty:
                    continue
                deltas = pair[f'{metric}_baseline'] - pair[f'{metric}_current']
                rows.append(
                    {
                        'forecast_source': source,
                        'forecast_version': version,
                        'baseline_version': baseline_version,
                        'metric': metric,
                        'n_questions': int(len(deltas)),
                        'mean_improvement': float(deltas.mean()),
                        'median_improvement': float(deltas.median()),
                        'std_improvement': float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                        'q1_improvement': float(deltas.quantile(0.25)),
                        'q3_improvement': float(deltas.quantile(0.75)),
                    }
                )

    improvement = pd.DataFrame(rows)
    if improvement.empty:
        return improvement
    improvement = improvement.sort_values(
        ['metric', 'forecast_version', 'forecast_source'],
        key=lambda s: s.map(_version_sort_key) if s.name == 'forecast_version' else s.astype(str),
    )
    return improvement.reset_index(drop=True)
