"""Run-directory layout and per-stage JSON dump helpers.

The orchestrator writes each stage's output and a running manifest to
``data/runs/{question_id}/{run_id}/`` so a crashed or interrupted run
still has partial artifacts for debugging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from bioscancast.filtering.models import ForecastQuestion
from bioscancast.insight.pipeline import InsightRunResult


def _json_default(obj: Any) -> Any:
    """Default JSON encoder for datetimes, dataclasses, and sets.

    Lifted from scripts/eval_insight_on_real_docs.py and extended to
    handle the set/frozenset values that live in FILTER_CONFIG.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not serializable: {type(obj).__name__}")


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def make_run_dir(out_root: Path, question_id: str, run_id: str) -> Path:
    """Create and return ``out_root/question_id/run_id/``."""
    run_dir = Path(out_root) / question_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_question(run_dir: Path, question: ForecastQuestion) -> None:
    _dump(run_dir / "question.json", asdict(question))


def save_search(run_dir: Path, results: Iterable[Any]) -> None:
    _dump(run_dir / "search.json", [asdict(r) for r in results])


def save_filtered(run_dir: Path, docs: Iterable[Any]) -> None:
    _dump(run_dir / "filtered.json", [asdict(d) for d in docs])


def save_documents(run_dir: Path, docs: Iterable[Any]) -> None:
    _dump(run_dir / "documents.json", [asdict(d) for d in docs])


def save_insight(run_dir: Path, result: InsightRunResult) -> None:
    """Serialize an InsightRunResult to insight.json.

    The dataclass nests InsightRecord and ChunkReference dataclasses;
    asdict handles both. budget_summary is already a dict.
    """
    payload = {
        "records": [asdict(r) for r in result.records],
        "budget_summary": result.budget_summary,
        "documents_processed": result.documents_processed,
        "documents_skipped": result.documents_skipped,
        "notes": list(result.notes),
    }
    _dump(run_dir / "insight.json", payload)


def save_manifest(run_dir: Path, manifest: dict) -> None:
    """Write the manifest. Called repeatedly — once before each stage and
    again after every stage completes — so a crashed run keeps the
    partial timings/config that did make it through.
    """
    _dump(run_dir / "manifest.json", manifest)
