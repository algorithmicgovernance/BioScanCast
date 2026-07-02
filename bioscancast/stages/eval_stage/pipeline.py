from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

from bioscancast.stages.eval_stage.compare import (
    compare_sources,
    compare_sources_by_question_type,
    compare_sources_over_time,
    relative_improvement_over_time,
)
from bioscancast.stages.eval_stage.loaders import load_forecasts, load_questions
from bioscancast.stages.eval_stage.scoring import (
    accuracy,
    log_score,
    multiclass_brier_score,
    normalized_entropy,
    ranked_probability_score,
    top_probability,
    true_probability,
)
from bioscancast.stages.eval_stage.visualisation import (
    plot_question_heatmap,
    plot_relative_improvement,
    plot_score_timeline_boxplots,
    plot_source_timeline,
)

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FORECASTS = [
    str(BASE_DIR / 'mock_forecasts' / 'human_forecasts.csv'),
    str(BASE_DIR / 'mock_forecasts' / 'llm_baseline_forecasts.csv'),
    str(BASE_DIR / 'mock_forecasts' / 'perplexity_forecasts.csv'),
]

def _current_run_date() -> str:
    return pd.Timestamp.today().normalize().strftime('%Y-%m-%d')


def _refresh_perplexity_forecasts(questions_path: str | Path = str(BASE_DIR / 'bioscancast_questions_resolved.csv')) -> Path:
    """Regenerate the Perplexity mock forecasts when the API is available.

    If the Perplexity API key is not configured, fall back to the existing cached
    CSV so the evaluation stage can still run offline.
    """
    output_path = BASE_DIR / 'mock_forecasts' / 'perplexity_forecasts.csv'
    try:
        from bioscancast.stages.eval_stage.generate_perplexity_forecasts import (
            DEFAULT_TEMPLATE_FORECASTS,
            build_forecasts,
        )
        rows = build_forecasts(questions_path, DEFAULT_TEMPLATE_FORECASTS, model='sonar-reasoning-pro')
    except Exception as exc:
        if output_path.exists():
            print(f'Perplexity refresh skipped ({exc}); using cached forecasts at {output_path}.')
            return output_path
        raise

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep=';', index=False, decimal=',', encoding='cp1252')
    print(f'Regenerated Perplexity forecasts at {output_path}')
    return output_path

def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _canonicalize_text(value) -> str:
    if pd.isna(value):
        return ''
    text = str(value)
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    return text.strip()


def _prepare_questions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = {'question_id', 'question_status', 'resolved_option'}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError('questions dataframe is missing required columns: ' + ', '.join(missing))

    df['question_id'] = df['question_id'].astype(str).str.strip()
    df['question_status'] = df['question_status'].astype(str).str.lower().str.strip()
    df['resolved_option'] = df['resolved_option'].apply(_canonicalize_text)

    if 'question_type' in df.columns:
        df['question_type'] = df['question_type'].astype(str).str.lower().str.strip()
    else:
        df['question_type'] = 'unknown'

    if 'topic' not in df.columns:
        df['topic'] = ''

    return df


def _infer_source_name(path: str | Path) -> str:
    stem = Path(path).stem.lower().strip()
    for suffix in ('_forecasts', '_forecast', '_mock', '_data'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or 'forecast'


def _prepare_forecasts(df: pd.DataFrame, source_name: str | None = None) -> pd.DataFrame:
    df = df.copy()

    required_cols = {'question_id', 'option', 'probability'}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError('forecasts dataframe is missing required columns: ' + ', '.join(missing))

    df['question_id'] = df['question_id'].astype(str).str.strip()
    df['option'] = df['option'].apply(_canonicalize_text)
    df['probability'] = pd.to_numeric(df['probability'], errors='coerce')

    if df['probability'].isna().any():
        bad_rows = df[df['probability'].isna()]
        raise ValueError(
            'Some forecast probabilities could not be parsed as numeric values. '
            'Problematic rows: ' + str(bad_rows.index.tolist())
        )

    if 'forecast_source' not in df.columns:
        df['forecast_source'] = source_name or 'forecast'
    else:
        df['forecast_source'] = df['forecast_source'].astype(str).str.strip()
        if source_name and (df['forecast_source'].nunique() == 1) and (df['forecast_source'].iloc[0] in {'', 'nan', 'none'}):
            df['forecast_source'] = source_name

    if 'forecast_version' not in df.columns:
        df['forecast_version'] = '1'
    else:
        df['forecast_version'] = df['forecast_version'].astype(str).str.strip()
        df.loc[df['forecast_version'].isin({'', 'nan', 'none'}), 'forecast_version'] = '1'

    if 'forecast_date' in df.columns:
        parsed_dates = pd.to_datetime(df['forecast_date'], errors='coerce')
        fallback_date = _current_run_date()
        df['forecast_date'] = parsed_dates.dt.strftime('%Y-%m-%d').fillna(fallback_date)
    else:
        df['forecast_date'] = _current_run_date()

    return df


def _get_resolved_option_for_group(group: pd.DataFrame, question_row: pd.Series) -> Tuple[str, str]:
    resolved_option = _canonicalize_text(question_row['resolved_option'])
    options = [_canonicalize_text(opt) for opt in group['option'].tolist()]

    if resolved_option in options:
        return resolved_option, ''

    lowered = resolved_option.lower()
    if lowered in {'', 'tbd', 'na', 'n/a', 'ambiguous'}:
        return '', f"unscorable_status:{lowered or 'empty'}"
    if lowered.startswith('resolved on '):
        return '', 'placeholder_resolution_text'

    return '', 'resolved_option_not_in_forecast_options'


def build_distribution(group: pd.DataFrame) -> Tuple[list[str], list[float]]:
    group = group.copy()
    if 'option_order' in group.columns:
        group = group.sort_values('option_order')

    options = [_canonicalize_text(opt) for opt in group['option'].tolist()]
    probabilities = group['probability'].astype(float).tolist()
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('Forecast probabilities must sum to a positive value.')
    probabilities = [p / total for p in probabilities]
    return options, probabilities


def score_all_forecasts(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    skipped = []

    grouped = merged_df.groupby(['question_id', 'forecast_source', 'forecast_version', 'forecast_date'], dropna=False)
    for (question_id, source, version, forecast_date), group in grouped:
        question_row = group.iloc[0]

        if str(question_row['question_status']).lower() != 'resolved':
            skipped.append(
                {
                    'question_id': question_id,
                    'forecast_source': source,
                    'forecast_version': version,
                    'forecast_date': forecast_date,
                    'skip_reason': f"question_status={question_row['question_status']}",
                }
            )
            continue

        options, probabilities = build_distribution(group)
        resolved_option, skip_reason = _get_resolved_option_for_group(group=group, question_row=question_row)
        if skip_reason:
            skipped.append(
                {
                    'question_id': question_id,
                    'forecast_source': source,
                    'forecast_version': version,
                    'forecast_date': forecast_date,
                    'skip_reason': skip_reason,
                }
            )
            continue

        if resolved_option not in options:
            skipped.append(
                {
                    'question_id': question_id,
                    'forecast_source': source,
                    'forecast_version': version,
                    'forecast_date': forecast_date,
                    'skip_reason': 'resolved_option_missing_after_normalization',
                }
            )
            continue

        true_index = options.index(resolved_option)
        brier = multiclass_brier_score(probabilities, true_index)
        logscore = log_score(probabilities, true_index)
        acc = accuracy(probabilities, true_index)
        rps = ranked_probability_score(probabilities, true_index)
        top_prob = top_probability(probabilities)
        norm_entropy = normalized_entropy(probabilities)
        true_prob = true_probability(probabilities, true_index)

        results.append(
            {
                'question_id': question_id,
                'topic': question_row.get('topic', ''),
                'question_type': question_row.get('question_type', 'unknown'),
                'forecast_source': source,
                'forecast_version': version,
                'forecast_date': forecast_date,
                'resolved_option': resolved_option,
                'brier_score': brier,
                'log_score': logscore,
                'accuracy': acc,
                'accuracy_error': 1 - acc,
                'rps': rps,
                'top_probability': top_prob,
                'normalized_entropy': norm_entropy,
                'true_probability': true_prob,
            }
        )

    return pd.DataFrame(results), pd.DataFrame(skipped)


def run_evaluation(
    forecasts_path: str | Sequence[str] | None = None,
    questions_path: str = str(BASE_DIR / 'bioscancast_questions_resolved.csv'),
) -> None:
    """End-to-end evaluation entry point."""
    _ensure_output_dir()

    if forecasts_path is None:
        forecast_paths = DEFAULT_FORECASTS
    elif isinstance(forecasts_path, (str, Path)):
        forecast_paths = [str(forecasts_path)]
    else:
        forecast_paths = [str(p) for p in forecasts_path]

    questions = _prepare_questions(load_questions(questions_path))
    _refresh_perplexity_forecasts(questions_path=questions_path)
    resolved_questions = questions[questions['question_status'] == 'resolved'].copy()
    ambiguous_questions = questions[questions['question_status'] == 'ambiguous'].copy()
    unresolved_questions = questions[questions['question_status'] == 'unresolved'].copy()

    if not ambiguous_questions.empty:
        ambiguous_questions.to_csv(OUTPUT_DIR / 'ambiguous_questions.csv', index=False)
    if not unresolved_questions.empty:
        unresolved_questions.to_csv(OUTPUT_DIR / 'unresolved_questions.csv', index=False)

    forecast_frames = []
    for forecast_path in forecast_paths:
        source_name = _infer_source_name(forecast_path)
        forecast_df = _prepare_forecasts(load_forecasts(forecast_path), source_name=source_name)
        forecast_frames.append(forecast_df)

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    merged = forecasts.merge(
        resolved_questions,
        on='question_id',
        how='inner',
        suffixes=('_forecast', '_question'),
    )

    results_df, skipped_df = score_all_forecasts(merged)

    results_path = OUTPUT_DIR / 'question_level_metrics.csv'
    skipped_path = OUTPUT_DIR / 'skipped_questions.csv'
    summary_path = OUTPUT_DIR / 'summary_metrics.csv'
    by_type_path = OUTPUT_DIR / 'summary_metrics_by_question_type.csv'
    timeline_summary_path = OUTPUT_DIR / 'summary_metrics_over_time.csv'
    improvement_path = OUTPUT_DIR / 'metric_improvement_over_time.csv'

    results_df.to_csv(results_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    if results_df.empty:
        print('No questions could be scored. Check the resolved_option values.')
        return

    summary_df = compare_sources(results_df)
    summary_df.to_csv(summary_path, index=False)

    by_type_df = compare_sources_by_question_type(results_df)
    by_type_df.to_csv(by_type_path, index=False)

    timeline_summary_df = compare_sources_over_time(results_df)
    timeline_summary_df.to_csv(timeline_summary_path, index=False)

    from bioscancast.stages.eval_stage.compare import metric_change_over_time
    improvement_df = metric_change_over_time(timeline_summary_df)
    improvement_df.to_csv(improvement_path, index=False)

    plot_score_timeline_boxplots(results_df, OUTPUT_DIR / 'score_timeline_boxplots.png')
    plot_source_timeline(timeline_summary_df, OUTPUT_DIR / 'source_timeline_summary.png')
    plot_relative_improvement(results_df, OUTPUT_DIR / 'improvement_vs_v1.png')
    plot_question_heatmap(results_df, OUTPUT_DIR / 'question_heatmap.png')

    print('\nEvaluation complete.')
    print(summary_df.to_string(index=False))

    if not timeline_summary_df.empty:
        print('\nTimeline summary saved to:')
        print(timeline_summary_path)

    if not skipped_df.empty:
        print('\nSkipped questions:')
        print(skipped_df.to_string(index=False))


if __name__ == '__main__':
    run_evaluation()
