from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from bioscancast.stages.eval_stage.compare import _version_sort_key, rank_sources_over_time, relative_improvement_over_time


METRICS = [
    ('brier_score', 'Brier score'),
    ('log_score', 'Log score'),
    ('accuracy_error', 'Accuracy error (1 - accuracy)'),
    ('rps', 'RPS'),
]


def _ensure_parent_dir(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _ordered_unique(values: Sequence[object]) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
    return seen


def _color_cycle(n: int) -> list[str]:
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4'])
    return [colors[i % len(colors)] for i in range(n)]


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('results_df is missing required columns: ' + ', '.join(missing))


def _version_axis(values: Sequence[object]) -> list[str]:
    return sorted(_ordered_unique(values), key=_version_sort_key)


def plot_score_timeline_boxplots(results_df: pd.DataFrame, output_path: str | Path) -> None:
    required = {'forecast_source', 'forecast_version', 'brier_score', 'log_score', 'accuracy_error', 'rps'}
    _require_columns(results_df, required)

    output_path = _ensure_parent_dir(output_path)
    versions = _version_axis(results_df['forecast_version'].tolist())
    sources = _ordered_unique(results_df['forecast_source'].tolist())
    colors = _color_cycle(len(sources))
    offsets = np.linspace(-0.25, 0.25, max(1, len(sources))) if len(sources) > 1 else np.array([0.0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, METRICS):
        base_positions = np.arange(len(versions), dtype=float)
        for source_idx, source in enumerate(sources):
            data = []
            positions = []
            for version_idx, version in enumerate(versions):
                mask = (results_df['forecast_source'].astype(str) == source) & (results_df['forecast_version'].astype(str) == version)
                vals = results_df.loc[mask, metric].dropna().astype(float).tolist()
                if vals:
                    data.append(vals)
                    positions.append(base_positions[version_idx] + offsets[source_idx])

            if data:
                bp = ax.boxplot(
                    data,
                    positions=positions,
                    widths=0.18 if len(sources) > 1 else 0.35,
                    patch_artist=True,
                    showmeans=True,
                    whis=(0, 100),
                )
                color = colors[source_idx]
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.28)
                    patch.set_edgecolor(color)
                for element in ['whiskers', 'caps', 'medians', 'means']:
                    for item in bp[element]:
                        item.set_color(color)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_xticks(np.arange(len(versions)))
        ax.set_xticklabels([str(v) for v in versions])
        ax.set_xlabel('Forecast version')

    legend_handles = [Patch(facecolor=colors[i], edgecolor=colors[i], label=sources[i], alpha=0.28) for i in range(len(sources))]
    if legend_handles:
        fig.legend(handles=legend_handles, title='Forecast source', loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle('Score distributions by source and version', y=0.972, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_source_timeline(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    required = {'forecast_version', 'forecast_source'} | {f'median_{metric}' for metric, _ in METRICS} | {f'q1_{metric}' for metric, _ in METRICS} | {f'q3_{metric}' for metric, _ in METRICS}
    _require_columns(summary_df, required)

    output_path = _ensure_parent_dir(output_path)
    versions = _version_axis(summary_df['forecast_version'].tolist())
    sources = _ordered_unique(summary_df['forecast_source'].tolist())
    colors = _color_cycle(len(sources))
    x = np.arange(len(versions), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, METRICS):
        for source_idx, source in enumerate(sources):
            source_df = summary_df[summary_df['forecast_source'].astype(str) == source].copy()
            source_df['forecast_version'] = source_df['forecast_version'].astype(str)
            source_df = source_df.set_index('forecast_version').reindex(versions)
            medians = source_df[f'median_{metric}'].astype(float).to_numpy()
            q1 = source_df[f'q1_{metric}'].astype(float).to_numpy()
            q3 = source_df[f'q3_{metric}'].astype(float).to_numpy()
            color = colors[source_idx]
            ax.plot(x, medians, marker='o', linewidth=2, label=source, color=color)
            ax.fill_between(x, q1, q3, alpha=0.18, color=color)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in versions])
        ax.set_xlabel('Forecast version')

    legend_handles = [Patch(facecolor=colors[i], edgecolor=colors[i], label=sources[i], alpha=0.18) for i in range(len(sources))]
    if legend_handles:
        fig.legend(handles=legend_handles, title='Forecast source', loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle('Median score timelines with IQR bands', y=0.972, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_relative_improvement(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = _ensure_parent_dir(output_path)
    improvement_df = relative_improvement_over_time(results_df)
    if improvement_df.empty:
        raise ValueError('No improvement data available to plot.')

    versions = _version_axis(improvement_df['forecast_version'].tolist())
    sources = _ordered_unique(improvement_df['forecast_source'].tolist())
    colors = _color_cycle(len(sources))
    x = np.arange(len(versions), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, METRICS):
        metric_df = improvement_df[improvement_df['metric'] == metric].copy()
        for source_idx, source in enumerate(sources):
            source_df = metric_df[metric_df['forecast_source'].astype(str) == source].copy()
            source_df['forecast_version'] = source_df['forecast_version'].astype(str)
            source_df = source_df.set_index('forecast_version').reindex(versions)
            medians = source_df['median_improvement'].astype(float).to_numpy()
            q1 = source_df['q1_improvement'].astype(float).to_numpy()
            q3 = source_df['q3_improvement'].astype(float).to_numpy()
            color = colors[source_idx]
            ax.plot(x, medians, marker='o', linewidth=2, label=source, color=color)
            ax.fill_between(x, q1, q3, alpha=0.18, color=color)

        ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
        ax.set_title(f'{label} improvement vs version 1')
        ax.set_ylabel('Improvement (positive = better)')
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in versions])
        ax.set_xlabel('Forecast version')

    legend_handles = [Patch(facecolor=colors[i], edgecolor=colors[i], label=sources[i], alpha=0.18) for i in range(len(sources))]
    if legend_handles:
        fig.legend(handles=legend_handles, title='Forecast source', loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle('Change relative to the first forecast version', y=0.972, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_question_heatmap(results_df: pd.DataFrame, output_path: str | Path, metric: str = 'brier_score') -> None:
    required = {'question_id', 'forecast_source', 'forecast_version', metric}
    _require_columns(results_df, required)

    output_path = _ensure_parent_dir(output_path)
    versions = _version_axis(results_df['forecast_version'].tolist())
    sources = _ordered_unique(results_df['forecast_source'].tolist())
    metric_label = dict(METRICS).get(metric, metric)

    fig, axes = plt.subplots(len(sources), 1, figsize=(12, max(4, 3.2 * len(sources))), sharex=True)
    if len(sources) == 1:
        axes = [axes]

    all_values = []
    ordered_frames = []
    for source in sources:
        source_df = results_df[results_df['forecast_source'].astype(str) == source].copy()
        pivot = source_df.pivot_table(index='question_id', columns='forecast_version', values=metric, aggfunc='mean')
        pivot = pivot.reindex(columns=versions)
        baseline = pivot[versions[0]] if versions else pd.Series(dtype=float)
        order = baseline.sort_values(ascending=False).index.tolist() if not baseline.empty else pivot.index.tolist()
        pivot = pivot.reindex(order)
        ordered_frames.append((source, pivot))
        if not pivot.empty:
            all_values.append(pivot.to_numpy(dtype=float))

    if all_values:
        stacked = np.concatenate([arr[np.isfinite(arr)] for arr in all_values if arr.size])
        vmin = float(np.nanmin(stacked))
        vmax = float(np.nanmax(stacked))
    else:
        vmin, vmax = 0.0, 1.0

    for ax, (source, pivot) in zip(axes, ordered_frames):
        values = pivot.to_numpy(dtype=float)
        im = ax.imshow(values, aspect='auto', interpolation='nearest', cmap='viridis_r', vmin=vmin, vmax=vmax)
        ax.set_title(source)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(q) for q in pivot.index])
        ax.set_xticks(np.arange(len(versions)))
        ax.set_xticklabels([str(v) for v in versions])
        ax.set_ylabel('Question')
        ax.grid(False)

    axes[-1].set_xlabel('Forecast version')
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label=metric_label)
    fig.suptitle(f'Question-level {metric_label} across versions', y=0.99, fontsize=16)
    fig.subplots_adjust(top=0.92, right=0.88)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_source_ranking_over_time(summary_df: pd.DataFrame, output_path: str | Path, metric_column: str = 'median_brier_score') -> None:
    output_path = _ensure_parent_dir(output_path)
    ranked = rank_sources_over_time(summary_df, metric_column=metric_column, ascending=True)
    if ranked.empty:
        raise ValueError('No ranking data available to plot.')

    versions = _version_axis(ranked['forecast_version'].tolist())
    sources = _ordered_unique(ranked['forecast_source'].tolist())
    matrix = pd.DataFrame(index=sources, columns=versions, dtype=float)
    for _, row in ranked.iterrows():
        matrix.loc[str(row['forecast_source']), str(row['forecast_version'])] = float(row['rank'])

    data = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap='YlGn_r', vmin=1, vmax=max(3, int(np.nanmax(data))))
    ax.set_xticks(np.arange(len(versions)))
    ax.set_xticklabels([str(v) for v in versions])
    ax.set_yticks(np.arange(len(sources)))
    ax.set_yticklabels(sources)
    ax.set_xlabel('Forecast version')
    ax.set_ylabel('Forecast source')
    ax.set_title('Source ranking over time (lower Brier score = better rank)')

    for i in range(len(sources)):
        for j in range(len(versions)):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f'{int(value)}', ha='center', va='center', color='black', fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label='Rank')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
