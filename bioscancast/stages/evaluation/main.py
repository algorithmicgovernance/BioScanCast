from __future__ import annotations

import argparse
from pathlib import Path

from bioscancast.stages.evaluation.pipeline import run_evaluation


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FORECASTS = [
    str(BASE_DIR / 'mock_forecasts' / 'human_forecasts.csv'),
    str(BASE_DIR / 'mock_forecasts' / 'llm_baseline_forecasts.csv'),
    str(BASE_DIR / 'mock_forecasts' / 'perplexity_forecasts.csv'),
]


def _collect_forecast_files(directory: str | Path) -> list[str]:
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Forecast directory not found: {directory}")
    files = sorted(p for p in path.glob("*.csv") if p.is_file())
    if not files:
        raise ValueError(f"No CSV forecasts found in {directory}")
    return [str(p) for p in files]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the BioScanCast evaluation pipeline.')
    parser.add_argument(
        '--forecasts',
        nargs='+',
        default=None,
        help='One or more forecast CSV files.',
    )
    parser.add_argument(
        '--forecast-dir',
        default=None,
        help='Directory containing forecast CSVs to evaluate together.',
    )
    parser.add_argument(
        '--questions',
        default=str(BASE_DIR / 'bioscancast_questions_resolved.csv'),
        help='Path to the questions CSV file.',
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for running the evaluation pipeline from the command line.
    """
    args = parse_args()
    if args.forecast_dir:
        forecasts = _collect_forecast_files(args.forecast_dir)
    elif args.forecasts:
        forecasts = args.forecasts
    else:
        forecasts = DEFAULT_FORECASTS
    run_evaluation(forecasts_path=forecasts, questions_path=args.questions)


if __name__ == '__main__':
    main()
