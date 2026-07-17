"""Custom scraper for USDA APHIS HPAI livestock detections.

The public APHIS dashboard is a client-rendered Tableau page; this scraper uses
headless browser automation to export the Tableau crosstab CSV and renders
compact analytical prose/tables so the regular HTML parser + insight extraction
pipeline can operate without additional parser changes.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from bioscancast.stages.extraction.config import ExtractionConfig
from bioscancast.stages.extraction.fetcher import FetchResult

logger = logging.getLogger(__name__)

TABLEAU_VIEW_URL = (
    "https://publicdashboards.dl.usda.gov/t/MRP_PUB/views/"
    "VS_Cattle_HPAIConfirmedDetections2024/HPAI2022ConfirmedDetections"
)

TableauFetcher = Callable[[str, ExtractionConfig], Optional[pd.DataFrame]]


def _embed_url(url: str) -> str:
    if ":embed=y" in url:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}:showVizHome=no&:embed=y"


def _read_csv_with_fallback(path: str) -> Optional[pd.DataFrame]:
    attempts = [
        {"encoding": "utf-16", "sep": "\t", "skiprows": 1},
        {"encoding": "utf-16-le", "sep": "\t", "skiprows": 1},
        {"encoding": "utf-8-sig", "sep": "\t", "skiprows": 1},
        {"encoding": "utf-8", "sep": "\t", "skiprows": 1},
        {"encoding": "utf-16"},
        {"encoding": "utf-16-le"},
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8"},
        {"encoding": "latin-1"},
    ]
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            logger.info("USDA Tableau: parsed CSV with options=%s", kwargs)
            return df
        except UnicodeDecodeError:
            logger.info("USDA Tableau: decode failed with options=%s", kwargs)
            continue
        except Exception as exc:
            logger.info("USDA Tableau: csv parse error with options=%s: %s", kwargs, exc)
            continue
    return None


def _fetch_tableau_dataframe(url: str, cfg: ExtractionConfig) -> Optional[pd.DataFrame]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        logger.info("USDA Tableau: Playwright unavailable: %s", exc)
        return None

    timeout_ms = int(max(cfg.fetch_timeout_seconds, 45.0) * 1000)

    try:
        with sync_playwright() as p:
            logger.info("USDA Tableau: launching headless browser")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            target_url = _embed_url(url)
            logger.info("USDA Tableau: navigating to %s", target_url)
            page.goto(_embed_url(url), wait_until="domcontentloaded", timeout=timeout_ms)

            logger.info("USDA Tableau: waiting for download toolbar button")
            for _ in range(200):
                if page.locator("button[data-tb-test-id='viz-viewer-toolbar-button-download']").count() > 0:
                    break
                page.wait_for_timeout(250)
            download_btn_count = page.locator("button[data-tb-test-id='viz-viewer-toolbar-button-download']").count()
            logger.info("USDA Tableau: download button count=%s", download_btn_count)
            if download_btn_count == 0:
                logger.info("USDA Tableau: download button not found before timeout")
                context.close()
                browser.close()
                return None

            logger.info("USDA Tableau: opening download menu")
            page.locator("button[data-tb-test-id='viz-viewer-toolbar-button-download']").first.click(timeout=20_000)
            logger.info("USDA Tableau: selecting Crosstab")
            page.locator("[data-tb-test-id='download-flyout-download-crosstab-MenuItem']").first.click(timeout=20_000)
            page.wait_for_timeout(1500)

            logger.info("USDA Tableau: selecting worksheet 'Table Details by Date'")
            page.locator("[data-tb-test-id^='sheet-thumbnail-']", has_text="Table Details by Date").first.click(timeout=20_000)
            logger.info("USDA Tableau: selecting CSV export format")
            page.locator("[data-tb-test-id='crosstab-options-dialog-radio-csv-Label']").first.click(timeout=20_000)

            downloads = []
            page.on("download", lambda d: downloads.append(d))
            logger.info("USDA Tableau: triggering export download")
            page.locator("[data-tb-test-id='export-crosstab-export-Button']").first.click(timeout=20_000)

            waited = 0
            step = 250
            while waited <= timeout_ms and not downloads:
                page.wait_for_timeout(step)
                waited += step
            logger.info("USDA Tableau: download events captured=%s after %sms", len(downloads), waited)

            if not downloads:
                logger.info("USDA Tableau: no download event received")
                context.close()
                browser.close()
                return None
            download = downloads[0]

            path = download.path()
            logger.info("USDA Tableau: download path resolved=%s", bool(path))
            if path is None:
                logger.info("USDA Tableau: download path unavailable")
                context.close()
                browser.close()
                return None

            logger.info("USDA Tableau: reading CSV from %s", path)
            df = _read_csv_with_fallback(path)
            context.close()
            browser.close()

            if df is None:
                logger.info("USDA Tableau: failed to parse CSV with fallback encodings")
                return None
            if df.empty:
                logger.info("USDA Tableau: CSV parsed but DataFrame empty")
                return None
            logger.info("USDA Tableau: CSV parsed rows=%s cols=%s", df.shape[0], df.shape[1])
            return df
    except Exception as exc:
        logger.info("USDA Tableau: fetch failed with exception: %s", exc)
        return None


def _resolve_column(df: pd.DataFrame, wanted: str) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    return by_lower.get(wanted.strip().lower())


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
    tableau_fetcher: TableauFetcher | None = None,
) -> FetchResult | None:
    cfg = config or ExtractionConfig()
    fetched_at = datetime.now(timezone.utc)

    df = (tableau_fetcher or _fetch_tableau_dataframe)(TABLEAU_VIEW_URL, cfg)
    if df is None or df.empty:
        return None

    df = df.copy()

    # Requested behavior: ignore first row.
    if df.shape[0] < 2:
        return None
    df = df.iloc[1:].copy()

    confirmed_col = _resolve_column(df, "Confirmed Diagnosis")
    state_col = _resolve_column(df, "State")
    if confirmed_col is None or state_col is None:
        return None

    if confirmed_col != "Confirmed Diagnosis":
        df = df.rename(columns={confirmed_col: "Confirmed Diagnosis"})
    if state_col != "State":
        df = df.rename(columns={state_col: "State"})

    # The crosstab renders dates like "14-Jul-26" (%d-%b-%y). Opt into
    # per-element parsing explicitly (format="mixed") — the same inference the
    # bare call already did, but without the "Could not infer format" warning.
    df["Confirmed Diagnosis"] = pd.to_datetime(
        df["Confirmed Diagnosis"], format="mixed", errors="coerce"
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
        f"<p>Tableau source URL: {html.escape(TABLEAU_VIEW_URL)}</p>"
        f"<p>Retrieved at: {fetched_at.isoformat()} | latest confirmed diagnosis date: {html.escape(latest)}.</p>"
        "<p>This summary is computed from Tableau workbook data. The first data row was ignored by design.</p>"
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
        url=TABLEAU_VIEW_URL,
        final_url=TABLEAU_VIEW_URL,
        status_code=200,
        content_type="text/html",
        content_bytes=rendered,
        fetched_at=fetched_at,
        error=None,
    )
