from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from bioscancast.stages.evaluation.compare import _timeline_sort_key, _version_sort_key, relative_improvement_over_time


METRICS = [
    ('brier_score', 'Brier score'),
    ('log_score', 'Log score'),
    ('accuracy_error', 'Accuracy error'),
    ('rps', 'RPS'),
]


def _ensure_parent_dir(output_path: str | Path) -> Path:
    """
    Make sure the destination folder exists before saving a figure.
    """
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


def _source_order_for_visibility(sources: Sequence[str]) -> list[str]:
    """Plot BioScanCast last so it stays visible on top of the other runs."""
    ordered = [str(source) for source in sources]
    bios = [s for s in ordered if s.lower() == 'bioscancast']
    others = [s for s in ordered if s.lower() != 'bioscancast']
    return others + bios


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('results_df is missing required columns: ' + ', '.join(missing))


def _timeline_column(df: pd.DataFrame) -> str:
    return 'forecast_date' if 'forecast_date' in df.columns else 'forecast_version'


def _timeline_axis(values: Sequence[object]) -> list[str]:
    values = list(values)
    if not values:
        return []
    parsed = pd.to_datetime(pd.Series(values), errors='coerce')
    if parsed.notna().any():
        return sorted(_ordered_unique(values), key=_timeline_sort_key)
    return sorted(_ordered_unique(values), key=_version_sort_key)


def _annotate_direction(ax, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, edgecolor='none'),
    )


def _latest_rows_for_source(results_df: pd.DataFrame) -> pd.DataFrame:
    time_col = _timeline_column(results_df)
    if time_col == 'forecast_date':
        df = results_df.copy()
        df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
        max_dates = df.groupby('forecast_source')['forecast_date'].transform('max')
        return df[df['forecast_date'].eq(max_dates)].copy()

    def _is_latest(series: pd.Series) -> pd.Series:
        ordered = series.astype(str).map(_version_sort_key)
        return ordered == ordered.max()

    mask = results_df.groupby('forecast_source')['forecast_version'].transform(_is_latest)
    return results_df[mask].copy()


def plot_score_timeline_boxplots(results_df: pd.DataFrame, output_path: str | Path) -> None:
    required = {'forecast_source', 'forecast_version', 'brier_score', 'log_score', 'accuracy_error', 'rps'}
    _require_columns(results_df, required)

    output_path = _ensure_parent_dir(output_path)
    plot_df = _latest_rows_for_source(results_df)
    sources = _ordered_unique(plot_df['forecast_source'].tolist())
    colors = _color_cycle(len(sources))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, METRICS):
        data = [plot_df.loc[plot_df['forecast_source'].astype(str) == source, metric].dropna().astype(float).tolist() for source in sources]
        positions = np.arange(len(sources), dtype=float)
        if any(len(vals) for vals in data):
            bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showmeans=True, whis=(0, 100))
            for idx, (patch, color) in enumerate(zip(bp['boxes'], colors)):
                patch.set_facecolor(to_rgba(color, 0.28))
                patch.set_edgecolor(color)
                patch.set_linewidth(1.2)
            for idx, color in enumerate(colors):
                for item in bp['whiskers'][2 * idx: 2 * idx + 2]:
                    item.set_color(color)
                    item.set_linewidth(1.2)
                for item in bp['caps'][2 * idx: 2 * idx + 2]:
                    item.set_color(color)
                    item.set_linewidth(1.2)
                bp['medians'][idx].set_color(color)
                bp['medians'][idx].set_linewidth(2.0)
                bp['means'][idx].set_markeredgecolor(color)
                bp['means'][idx].set_markerfacecolor(color)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticks(positions)
        ax.set_xticklabels(sources)
        ax.grid(True, axis='y', alpha=0.2)
        _annotate_direction(ax, 'Lower is better')

    legend_handles = [Patch(facecolor=to_rgba(colors[i], 0.28), edgecolor=colors[i], label=sources[i]) for i in range(len(sources))]
    if legend_handles:
        fig.legend(handles=legend_handles, title='Forecast source', loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle('Latest score distributions by source', y=0.972, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_source_timeline(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    timeline_col = _timeline_column(summary_df)
    required = {timeline_col, 'forecast_source'} | {f'mean_{metric}' for metric, _ in METRICS} | {f'std_{metric}' for metric, _ in METRICS} | {'n_questions'}
    _require_columns(summary_df, required)

    output_path = _ensure_parent_dir(output_path)
    versions = _timeline_axis(summary_df[timeline_col].tolist())
    sources = _source_order_for_visibility(_ordered_unique(summary_df['forecast_source'].tolist()))
    colors = _color_cycle(len(sources))
    x = np.arange(len(versions), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, METRICS):
        for source_idx, source in enumerate(sources):
            source_df = summary_df[summary_df['forecast_source'].astype(str) == source].copy()
            source_df[timeline_col] = source_df[timeline_col].astype(str)
            source_df = source_df.set_index(timeline_col).reindex(versions)
            means = source_df[f'mean_{metric}'].astype(float).to_numpy()
            stds = source_df[f'std_{metric}'].astype(float).to_numpy()
            counts = source_df['n_questions'].astype(float).to_numpy()
            sem = np.divide(stds, np.sqrt(counts), out=np.zeros_like(stds, dtype=float), where=np.isfinite(stds) & np.isfinite(counts) & (counts > 0))
            lower = means - sem
            upper = means + sem
            color = colors[source_idx]
            is_bio = source.lower() == 'bioscancast'
            linewidth = 3.0 if is_bio else 2.0
            markersize = 7 if is_bio else 5
            zorder = 4 if is_bio else 2 + source_idx
            ax.plot(x, means, marker='o', markersize=markersize, linewidth=linewidth, label=source, color=color, zorder=zorder)
            ax.fill_between(x, lower, upper, alpha=0.18, color=color, zorder=1)

        # Mark the best source at each timepoint with a single star.
        for idx, version in enumerate(versions):
            best_source = None
            best_val = None
            for source_idx, source in enumerate(sources):
                source_df = summary_df[summary_df['forecast_source'].astype(str) == source].copy()
                source_df[timeline_col] = source_df[timeline_col].astype(str)
                row = source_df[source_df[timeline_col] == str(version)]
                if row.empty:
                    continue
                val = float(row.iloc[0][f'mean_{metric}'])
                if not np.isfinite(val):
                    continue
                if best_val is None or val < best_val:
                    best_val = val
                    best_source = source_idx
            if best_source is not None and best_val is not None:
                ax.scatter([x[idx]], [best_val], s=120, marker='*', color=colors[best_source], edgecolor='black', linewidth=0.5, zorder=6)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in versions], rotation=20, ha='right')
        ax.set_xlabel('Forecast date' if timeline_col == 'forecast_date' else 'Forecast version')
        _annotate_direction(ax, 'Lower is better')

    legend_handles = [Patch(facecolor=colors[i], edgecolor=colors[i], label=sources[i], alpha=0.18) for i in range(len(sources))]
    if legend_handles:
        fig.legend(handles=legend_handles, title='Forecast source', loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle('Mean score timelines with standard error bands', y=0.972, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_relative_improvement(results_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = _ensure_parent_dir(output_path)
    from bioscancast.stages.evaluation.compare import compare_sources_over_time, metric_change_over_time

    timeline_summary = compare_sources_over_time(results_df)
    improvement_df = metric_change_over_time(timeline_summary)
    if improvement_df.empty:
        raise ValueError('No improvement data available to plot.')

    timeline_col = 'forecast_date' if 'forecast_date' in improvement_df.columns else 'forecast_version'
    versions = _timeline_axis(improvement_df[timeline_col].tolist())
    sources = _source_order_for_visibility(_ordered_unique(improvement_df['forecast_source'].tolist()))
    colors = _color_cycle(len(sources))
    x = np.arange(len(versions), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, METRICS):
        metric_df = improvement_df[improvement_df['metric'] == metric].copy()
        for source_idx, source in enumerate(sources):
            source_df = metric_df[metric_df['forecast_source'].astype(str) == source].copy()
            source_df[timeline_col] = source_df[timeline_col].astype(str)
            source_df = source_df.set_index(timeline_col).reindex(versions)
            values = source_df['mean_improvement'].astype(float).to_numpy()
            color = colors[source_idx]
            is_bio = source.lower() == 'bioscancast'
            linewidth = 3.0 if is_bio else 2.0
            markersize = 7 if is_bio else 5
            zorder = 4 if is_bio else 2 + source_idx
            ax.plot(x, values, marker='o', markersize=markersize, linewidth=linewidth, label=source, color=color, zorder=zorder)

        # Single winning source marker per timepoint: the lowest change value wins
        for idx, version in enumerate(versions):
            best_source = None
            best_val = None
            for source_idx, source in enumerate(sources):
                row = metric_df[(metric_df['forecast_source'].astype(str) == source) & (metric_df[timeline_col].astype(str) == str(version))]
                if row.empty:
                    continue
                val = float(row.iloc[0]['mean_improvement'])
                if not np.isfinite(val):
                    continue
                if best_val is None or val < best_val:
                    best_val = val
                    best_source = source_idx
            if best_source is not None and best_val is not None:
                ax.scatter([x[idx]], [best_val], s=120, marker='*', color=colors[best_source], edgecolor='black', linewidth=0.5, zorder=6)

        ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
        ax.set_title(f'{label} change vs first date')
        ax.set_ylabel('Change (lower is better)')
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in versions], rotation=20, ha='right')
        ax.set_xlabel('Forecast date' if timeline_col == 'forecast_date' else 'Forecast version')
        _annotate_direction(ax, 'Lower change is better')

    legend_handles = [Patch(facecolor=colors[i], edgecolor=colors[i], label=sources[i], alpha=0.18) for i in range(len(sources))]
    if legend_handles:
        fig.legend(handles=legend_handles, title='Forecast source', loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=min(4, len(legend_handles)), frameon=False)
    fig.suptitle('Change relative to the first forecast date', y=0.972, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_question_heatmap(results_df: pd.DataFrame, output_path: str | Path, metric: str = 'brier_score') -> None:
    timeline_col = _timeline_column(results_df)
    required = {'question_id', 'forecast_source', timeline_col, metric}
    _require_columns(results_df, required)

    output_path = _ensure_parent_dir(output_path)
    versions = _timeline_axis(results_df[timeline_col].tolist())
    sources = _ordered_unique(results_df['forecast_source'].tolist())
    metric_label = dict(METRICS).get(metric, metric)

    fig, axes = plt.subplots(len(sources), 1, figsize=(12, max(4, 3.2 * len(sources))), sharex=True)
    if len(sources) == 1:
        axes = [axes]

    all_values = []
    ordered_frames = []
    for source in sources:
        source_df = results_df[results_df['forecast_source'].astype(str) == source].copy()
        pivot = source_df.pivot_table(index='question_id', columns=timeline_col, values=metric, aggfunc='mean')
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
        ax.set_xticklabels([str(v) for v in versions], rotation=20, ha='right')
        ax.set_ylabel('Question')
        ax.grid(False)

    axes[-1].set_xlabel('Forecast date' if timeline_col == 'forecast_date' else 'Forecast version')
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label=metric_label)
    fig.suptitle(f'Question-level {metric_label} across time', y=0.99, fontsize=16)
    fig.subplots_adjust(top=0.92, right=0.88)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
