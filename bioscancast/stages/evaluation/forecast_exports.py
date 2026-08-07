from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("data/forecast_outputs")
OUTPUT_FILENAMES = {
    "runs_bfg_summer_2026_no_history": "bfg_summer_2026_no_history.csv",
    "runs_bfg_summer_2026_with_history": "bfg_summer_2026_with_history.csv",
}
OUTPUT_COLUMNS = [
    "question_id",
    "forecast_source",
    "forecast_version",
    "option",
    "probability",
    "forecast_date",
]


def _normalize_forecast_date(run_id: str) -> str:
    if not run_id:
        return ""
    parts = str(run_id).split("_")
    if len(parts) >= 1 and parts[0].isdigit() and len(parts[0]) == 8:
        return f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:8]}"
    return str(run_id)


def _normalize_forecast_version(run_id: str) -> str:
    return _normalize_forecast_date(run_id)


def _load_existing_rows(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    with output_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _output_path_for_run_root(repo_root: Path, out_root: Path) -> Path:
    filename = OUTPUT_FILENAMES.get(out_root.name)
    if filename is None:
        raise ValueError(f"Unsupported forecast output root: {out_root}")
    return repo_root / OUTPUT_DIR / filename


def _record_to_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if is_dataclass(record):
        return asdict(record)
    if hasattr(record, "__dict__"):
        return dict(vars(record))
    raise TypeError(f"Unsupported forecast record type: {type(record)!r}")


def append_forecast_rows_to_output(
    out_root: Path | str,
    question_id: str,
    run_id: str,
    forecast_result: Any,
    *,
    repo_root: Path | None = None,
) -> Path:
    out_root_path = Path(out_root)
    if repo_root is not None:
        repo_root_path = Path(repo_root)
    else:
        repo_root_path = Path.cwd()

    output_path = _output_path_for_run_root(repo_root_path, out_root_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = _load_existing_rows(output_path)
    rows_to_append: list[dict[str, Any]] = []

    run_dir = out_root_path / question_id / run_id
    question_path = run_dir / "question.json"
    if question_path.exists():
        try:
            question_payload = json.loads(question_path.read_text(encoding="utf-8"))
            if question_payload.get("question_id"):
                question_id = question_payload["question_id"]
        except Exception:
            pass

    records = getattr(forecast_result, "records", None) or []
    if not records:
        return output_path

    for record in records:
        record_data = _record_to_mapping(record)
        row = {
            "question_id": record_data.get("question_id") or question_id,
            "forecast_source": record_data.get("forecast_source") or "bioscancast",
            "forecast_version": record_data.get("forecast_version") or _normalize_forecast_version(run_id),
            "option": record_data.get("option") or "",
            "probability": record_data.get("probability"),
            "forecast_date": _normalize_forecast_date(run_id),
        }
        rows_to_append.append(row)

    if not rows_to_append:
        return output_path

    combined_rows = existing_rows + rows_to_append
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_COLUMNS})

    return output_path
