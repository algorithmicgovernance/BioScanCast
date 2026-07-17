"""Custom scraper for PAHO ARBO Portal Oropouche weekly CSV.

The PAHO Oropouche portal is dashboard-backed. This scraper pulls the downloadable
underlying CSV and renders computed weekly/country analytics as compact HTML so
standard extraction + insight stages can consume the information.
"""

from __future__ import annotations

import html
import io
import re
import unicodedata
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from curl_cffi import requests as curl_requests

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

CSV_URL = (
    "https://phip.paho.org/vizql/w/AME_OROV_Cases/v/Oropouche/vudcsv/"
    "sessions/21F758F6EAB9431B932EB51F61EBDE01-1:0/views/"
    "10956961424773455462_12644916002446576356"
    "?underlying_table_id=FZ_AME_OROV_Cases_SE_12E30DDB8F874DC6AE5888461B08AB74"
    "&underlying_table_caption=Full%20Data"
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


def _normalize_col(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower().strip()
    folded = re.sub(r"[^a-z0-9]+", "_", folded)
    return folded.strip("_")


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str, str, str] | None:
    mapping = {_normalize_col(c): c for c in df.columns}

    year = mapping.get("ano")
    week = mapping.get("semanas_epi")
    confirmed = mapping.get("week_lab_confirmed")
    country = mapping.get("pais")

    if not all((year, week, confirmed, country)):
        return None
    return year, week, confirmed, country


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
    std = desc["std"] if pd.notna(desc["std"]) else 0.0
    return (
        f"n={int(desc['count'])}, mean={desc['mean']:.2f}, std={std:.2f}, "
        f"min={desc['min']:.0f}, median={series.median():.0f}, max={desc['max']:.0f}"
    )


def _render_counts_table(title: str, counts: pd.Series, key_header: str) -> str:
    rows = []
    for idx, value in counts.items():
        rows.append(f"<tr><td>{html.escape(str(idx))}</td><td>{int(value)}</td></tr>")
    body = "".join(rows) if rows else "<tr><td colspan='2'>No data</td></tr>"
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        f"<thead><tr><th>{html.escape(key_header)}</th><th>sum_week_lab_confirmed</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _render_model_section(
    title: str,
    counts: pd.Series,
    *,
    linear: dict[str, float] | None,
    exp: dict[str, float] | None,
) -> str:
    lines = [f"<h3>{html.escape(title)}</h3>"]
    lines.append(f"<p>Input points: {counts.shape[0]}.</p>")
    if linear is None:
        lines.append("<p>Linear model: unavailable (insufficient data).</p>")
    else:
        lines.append(
            "<p>Linear model y = intercept + slope*x: "
            f"intercept={linear['intercept']:.6f}, slope={linear['slope']:.6f}, R^2={linear['r2']:.6f}.</p>"
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


def _render_country_summary(df: pd.DataFrame, country_col: str, value_col: str) -> str:
    if df.empty:
        return "<h3>Per-country summary</h3><p>No country data available.</p>"

    g = df.groupby(country_col)[value_col].sum().sort_values(ascending=False)
    summary = g.describe()
    std = summary["std"] if pd.notna(summary["std"]) else 0.0

    rows = []
    for country, value in g.items():
        rows.append(
            f"<tr><td>{html.escape(str(country))}</td><td>{int(value)}</td></tr>"
        )

    return (
        "<h3>Per-country summary</h3>"
        f"<p>Affected states/countries (Pais values): {int(g.shape[0])}. "
        f"Per-country cumulative summary: n={int(summary['count'])}, mean={summary['mean']:.2f}, "
        f"std={std:.2f}, min={summary['min']:.0f}, median={g.median():.0f}, max={summary['max']:.0f}. "
        f"Cumulative confirmed count across all Pais: {int(g.sum())}.</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<thead><tr><th>Pais</th><th>cumulative_week_lab_confirmed</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


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

    cols = _resolve_columns(df)
    if cols is None:
        return None
    year_col, week_col, confirmed_col, country_col = cols

    df = df[[year_col, week_col, confirmed_col, country_col]].copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[week_col] = pd.to_numeric(df[week_col], errors="coerce")

    value_text = (
        df[confirmed_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[confirmed_col] = pd.to_numeric(value_text, errors="coerce").fillna(0)

    df = df.dropna(subset=[year_col, week_col]).copy()
    if df.empty:
        return None

    df[year_col] = df[year_col].astype(int)
    df[week_col] = df[week_col].astype(int)
    df = df[(df[week_col] >= 1) & (df[week_col] <= 53)].copy()
    if df.empty:
        return None

    df["week_start"] = [
        datetime.fromisocalendar(int(y), int(w), 1).date()
        for y, w in zip(df[year_col], df[week_col])
    ]

    if as_of_date is not None:
        cutoff = as_of_date.astimezone(timezone.utc).date()
        df = df[df["week_start"] <= cutoff]
        if df.empty:
            return None

    df["year_week"] = [f"{int(y)}-W{int(w):02d}" for y, w in zip(df[year_col], df[week_col])]

    weekly_counts = (
        df.groupby(["week_start", "year_week"], as_index=False)[confirmed_col]
        .sum()
        .sort_values("week_start")
    )
    if weekly_counts.empty:
        return None

    weekly_series = pd.Series(
        weekly_counts[confirmed_col].to_numpy(),
        index=weekly_counts["year_week"].tolist(),
    )

    recent_24 = weekly_counts.tail(24)
    recent_24_series = pd.Series(
        recent_24[confirmed_col].to_numpy(),
        index=recent_24["year_week"].tolist(),
    )

    lin = _fit_linear(recent_24_series)
    exp = _fit_exponential(recent_24_series)

    country_week = (
        df.groupby(["year_week", country_col], as_index=False)[confirmed_col]
        .sum()
        .sort_values(["year_week", country_col])
    )

    cumulative_all = int(weekly_series.sum())
    latest_week = str(weekly_counts["year_week"].iloc[-1])

    rendered = (
        "<html><head><meta charset='utf-8'>"
        "<title>PAHO Oropouche portal - weekly CSV analytics</title>"
        "</head><body>"
        "<h1>PAHO Oropouche weekly analytics snapshot</h1>"
        f"<p>Source portal URL: {html.escape(url)}</p>"
        f"<p>CSV source URL: {html.escape(CSV_URL)}</p>"
        f"<p>Retrieved at: {fetched_at.isoformat()} | latest year_week: {html.escape(latest_week)}.</p>"
        "<h2>Weekly counts from Año + Semanas Epi</h2>"
        f"<p>Summary statistics (weekly summed Week Lab Confirmed): {_fmt_stats(weekly_series)}. "
        f"Cumulative confirmed count across all weeks: {cumulative_all}.</p>"
        f"{_render_counts_table('Counts by year_week', weekly_series, 'year_week')}"
        f"{_render_model_section('Model fit on past 24 weeks (approx past 6 months)', recent_24_series, linear=lin, exp=exp)}"
        "<h2>Country/state grouping from year_week + Pais</h2>"
        f"<p>Grouped rows (year_week x Pais): {int(country_week.shape[0])}.</p>"
        f"{_render_country_summary(df, country_col, confirmed_col)}"
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
