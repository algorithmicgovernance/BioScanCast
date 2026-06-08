from __future__ import annotations

import argparse
from pathlib import Path

from bioscancast.stages.eval_stage.pipeline import run_evaluation


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FORECASTS = [
    str(BASE_DIR / 'mock_forecasts' / 'human_forecasts.csv'),
    str(BASE_DIR / 'mock_forecasts' / 'bioscancast_forecasts.csv'),
    str(BASE_DIR / 'mock_forecasts' / 'llm_baseline_forecasts.csv'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the BioScanCast evaluation pipeline.')
    parser.add_argument(
        '--forecasts',
        nargs='+',
        default=DEFAULT_FORECASTS,
        help='One or more forecast CSV files. Use forecast_version to place repeated runs on the timeline.',
    )
    parser.add_argument(
        '--questions',
        default=str(BASE_DIR / 'bioscancast_questions.csv'),
        help='Path to the questions CSV file.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(forecasts_path=args.forecasts, questions_path=args.questions)


if __name__ == '__main__':
    main()
