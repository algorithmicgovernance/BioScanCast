"""Custom scraper for USDA APHIS HPAI livestock detections.

The public APHIS dashboard is a client-rendered Tableau page; this scraper reads
its downloadable CSV export and renders compact analytical prose/tables so the
regular HTML parser + insight extraction pipeline can operate without additional
CSV-specific parser changes.
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from curl_cffi import requests as curl_requests

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

CSV_URL = (
    "https://publicdashboards.dl.usda.gov/vizql/t/MRP_PUB/"
    "w/VS_Cattle_HPAIConfirmedDetections2024/v/HPAI2022ConfirmedDetections/"
    "tempfile/sessions/89200926388D46C4A9F17603C7900132-1:0/"
    "?key=2503115340&keepfile=yes&attachment=yes"
)

CsvFetcher = Callable[[str, ExtractionConfig], Optional[str]]


def _fetch_csv_text(url: str, cfg: ExtractionConfig) -> Optional[str]:
    try:
        resp = curl_requests.get(
            url,
            timeout=max(cfg.fetch_timeout_seconds, 30.0),
            impersonate=cfg.impersonate,
            allow_redirects=True,
        )
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    text = (resp.text or "").strip()
    return text or None


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))


def _fit_linear(counts: pd.Series) -> dict[str, float] | None:
    if counts.shape[0] < 2:
        return None
    y = counts.to_numpy(dtype=float)
    x = np.arange(y.shape[0], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": _r2(y, yhat),
    }


def _fit_exponential(counts: pd.Series) -> dict[str, float] | None:
    if counts.shape[0] < 2:
        return None
    y = counts.to_numpy(dtype=float)
    x = np.arange(y.shape[0], dtype=float)

    # Log-linear fit requires strictly positive observations.
    mask = y > 0
    if int(np.sum(mask)) < 2:
        return None

    x_pos = x[mask]
    y_pos = y[mask]

    b, ln_a = np.polyfit(x_pos, np.log(y_pos), 1)
    a = float(np.exp(ln_a))
    yhat = a * np.exp(b * x_pos)

    return {
        "a": a,
        "b": float(b),
        "r2": _r2(y_pos, yhat),
    }


def _fmt_stats(series: pd.Series) -> str:
    if series.empty:
        return "n=0"
    desc = series.describe()
    return (
        f"n={int(desc['count'])}, mean={desc['mean']:.2f}, std={desc['std'] if pd.notna(desc['std']) else 0.0:.2f}, "
        f"min={desc['min']:.0f}, median={series.median():.0f}, max={desc['max']:.0f}"
    )


def _render_counts_table(title: str, counts: pd.Series, key_header: str) -> str:
    rows = []
    for idx, value in counts.items():
        rows.append(
            f"<tr><td>{html.escape(str(idx))}</td><td>{int(value)}</td></tr>"
        )
    body = "".join(rows) if rows else "<tr><td colspan='2'>No data</td></tr>"
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        f"<thead><tr><th>{html.escape(key_header)}</th><th>count</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _render_state_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "<h3>Per-state summary</h3><p>No state data available.</p>"

    g = df.groupby("State").size().sort_values(ascending=False)
    summary = g.describe()

    rows = []
    for state, value in g.items():
        rows.append(f"<tr><td>{html.escape(str(state))}</td><td>{int(value)}</td></tr>")

    return (
        "<h3>Per-state summary</h3>"
        f"<p>State-case-count statistics: n={int(summary['count'])}, "
        f"mean={summary['mean']:.2f}, std={summary['std'] if pd.notna(summary['std']) else 0.0:.2f}, "
        f"min={summary['min']:.0f}, median={g.median():.0f}, max={summary['max']:.0f}.</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<thead><tr><th>State</th><th>case_count</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_first_case_by_state(df: pd.DataFrame) -> str:
    if df.empty:
        return "<h3>Affected states and first detected dates</h3><p>No state data available.</p>"

    first_by_state = (
        df.groupby("State")["Confirmed Diagnosis"]
        .min()
        .sort_values()
    )

    rows = []
    for state, dt in first_by_state.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(state))}</td>"
            f"<td>{html.escape(dt.date().isoformat())}</td>"
            "</tr>"
        )

    return (
        "<h3>Affected states and first detected dates</h3>"
        f"<p>Total affected states: {int(first_by_state.shape[0])}.</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<thead><tr><th>State</th><th>first_confirmed_diagnosis_date</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_model_section(
    title: str,
    counts: pd.Series,
    *,
    linear: dict[str, float] | None,
    exp: dict[str, float] | None,
) -> str:
    lines = [f"<h3>{html.escape(title)}</h3>"]
    # Repeat the section title in prose so it survives heading-only drops in
    # some HTML extraction paths and remains quotable for insight extraction.
    lines.append(
        f"<p>Section title: {html.escape(title)}. "
        f"Input points: {counts.shape[0]}.</p>"
    )

    if linear is None:
        lines.append("<p>Linear model: unavailable (insufficient data).</p>")
    else:
        lines.append(
            "<p>Linear model y = intercept + slope*x: "
            f"intercept={linear['intercept']:.6f}, slope={linear['slope']:.6f}, "
            f"R^2={linear['r2']:.6f}.</p>"
        )

    if exp is None:
        lines.append(
            "<p>Exponential model y = a*exp(b*x): unavailable "
            "(insufficient strictly-positive points).</p>"
        )
    else:
        lines.append(
            "<p>Exponential model y = a*exp(b*x): "
            f"a={exp['a']:.6f}, b={exp['b']:.6f}, R^2={exp['r2']:.6f}.</p>"
        )

    return "".join(lines)


def fetch(
    url: str,
    *,
    config: ExtractionConfig | None = None,
    as_of_date: datetime | None = None,
    region: str | None = None,
    question_text: str | None = None,
    csv_fetcher: CsvFetcher | None = None,
) -> FetchResult | None:
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)

    text = (csv_fetcher or _fetch_csv_text)(CSV_URL, cfg)
    if not text:
        return None

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return None

    # Requested behavior: ignore first row.
    if df.shape[0] < 2:
        return None
    df = df.iloc[1:].copy()

    if "Confirmed Diagnosis" not in df.columns or "State" not in df.columns:
        return None

    df["Confirmed Diagnosis"] = pd.to_datetime(
        df["Confirmed Diagnosis"], errors="coerce"
    )
    df = df.dropna(subset=["Confirmed Diagnosis"]).copy()
    if df.empty:
        return None

    if as_of_date is not None:
        cutoff = as_of_date.astimezone(timezone.utc).replace(tzinfo=None)
        df = df[df["Confirmed Diagnosis"] <= cutoff]
        if df.empty:
            return None

    # Normalize to date only for groupings.
    df["diagnosis_date"] = df["Confirmed Diagnosis"].dt.date
    df["month"] = df["Confirmed Diagnosis"].dt.to_period("M").astype(str)

    monthly_counts = df.groupby("month").size().sort_index()
    daily_counts = df.groupby("diagnosis_date").size().sort_index()
    cumulative_case_count = int(df.shape[0])

    if monthly_counts.empty or daily_counts.empty:
        return None

    month_window = monthly_counts.iloc[-6:]
    day_window = daily_counts.iloc[-30:]

    lin_month = _fit_linear(month_window)
    exp_month = _fit_exponential(month_window)
    lin_day = _fit_linear(day_window)
    exp_day = _fit_exponential(day_window)

    latest = str(daily_counts.index.max())

    rendered = (
        "<html><head><meta charset='utf-8'>"
        "<title>USDA APHIS HPAI Confirmed Cases in Livestock - CSV analytics</title>"
        "</head><body>"
        "<h1>USDA APHIS HPAI Confirmed Cases in Livestock - analytics snapshot</h1>"
        f"<p>Source dashboard URL: {html.escape(url)}</p>"
        f"<p>CSV source URL: {html.escape(CSV_URL)}</p>"
        f"<p>Retrieved at: {fetched_at.isoformat()} | latest confirmed diagnosis date: {html.escape(latest)}.</p>"
        "<p>This summary is computed from the downloadable APHIS CSV. The first CSV row was ignored by design.</p>"
        "<h2>Cumulative and state coverage summary</h2>"
        f"<p>Cumulative confirmed cases in livestock (from CSV rows): {cumulative_case_count}.</p>"
        f"{_render_first_case_by_state(df)}"
        "<h2>Monthly counts from Confirmed Diagnosis</h2>"
        f"<p>Summary statistics (monthly case counts): {_fmt_stats(monthly_counts)}.</p>"
        f"{_render_counts_table('Counts by month', monthly_counts, 'month')}"
        f"{_render_model_section('Model fit on past 6 months (monthly counts)', month_window, linear=lin_month, exp=exp_month)}"
        "<h2>Daily counts from Confirmed Diagnosis</h2>"
        f"<p>Summary statistics (daily case counts): {_fmt_stats(daily_counts)}.</p>"
        f"{_render_counts_table('Counts by day', daily_counts, 'date')}"
        f"{_render_model_section('Model fit on past 30 days (daily counts)', day_window, linear=lin_day, exp=exp_day)}"
        "<h2>State-level counts from Confirmed Diagnosis + State</h2>"
        f"{_render_state_summary(df)}"
        "</body></html>"
    ).encode("utf-8")

    return FetchResult(
        url=CSV_URL,
        final_url=CSV_URL,
        status_code=200,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )
