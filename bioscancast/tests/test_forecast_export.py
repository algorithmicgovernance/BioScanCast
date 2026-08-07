import csv
from pathlib import Path
from types import SimpleNamespace

from bioscancast.stages.evaluation.schemas import ForecastRecord
from bioscancast.stages.evaluation.forecast_exports import append_forecast_rows_to_output


def _forecast_result(option: str, probability: float) -> SimpleNamespace:
    return SimpleNamespace(
        records=[
            {
                "question_id": "bfg_q1",
                "forecast_source": "bioscancast",
                "option": option,
                "probability": probability,
                "forecast_version": None,
            }
        ]
    )


def test_append_forecast_rows_to_with_history_csv(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_root = repo_root / "data" / "runs_bfg_summer_2026_with_history"
    out_root.mkdir(parents=True, exist_ok=True)

    output_path = append_forecast_rows_to_output(
        out_root,
        "bfg_q1",
        "20260730_with_history",
        _forecast_result("YES", 0.6),
        repo_root=repo_root,
    )

    assert output_path == repo_root / "data" / "forecast_outputs" / "bfg_summer_2026_with_history.csv"
    assert output_path.exists()

    append_forecast_rows_to_output(
        out_root,
        "bfg_q1",
        "20260729_234607",
        _forecast_result("NO", 0.4),
        repo_root=repo_root,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["forecast_date"] == "2026-07-30"
    assert rows[1]["forecast_date"] == "2026-07-29"
    assert rows[0]["forecast_version"] == "2026-07-30"
    assert rows[1]["forecast_version"] == "2026-07-29"


def test_append_forecast_rows_to_no_history_csv(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_root = repo_root / "data" / "runs_bfg_summer_2026_no_history"
    out_root.mkdir(parents=True, exist_ok=True)

    output_path = append_forecast_rows_to_output(
        out_root,
        "bfg_q1",
        "20260805_120000",
        _forecast_result("YES", 0.6),
        repo_root=repo_root,
    )

    assert output_path == repo_root / "data" / "forecast_outputs" / "bfg_summer_2026_no_history.csv"
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["forecast_date"] == "2026-08-05"
    assert rows[0]["forecast_version"] == "2026-08-05"


def test_append_forecast_rows_accepts_forecast_record_objects(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_root = repo_root / "data" / "runs_bfg_summer_2026_no_history"
    out_root.mkdir(parents=True, exist_ok=True)

    forecast_result = SimpleNamespace(
        records=[
            ForecastRecord(
                question_id="bfg_q1",
                forecast_source="bioscancast",
                option="YES",
                probability=0.6,
            )
        ]
    )

    output_path = append_forecast_rows_to_output(
        out_root,
        "bfg_q1",
        "20260805_120000",
        forecast_result,
        repo_root=repo_root,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["question_id"] == "bfg_q1"
    assert rows[0]["forecast_source"] == "bioscancast"
    assert rows[0]["option"] == "YES"
