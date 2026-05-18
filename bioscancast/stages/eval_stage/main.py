from __future__ import annotations

import argparse
from pathlib import Path

from bioscancast.stages.eval_stage.pipeline import run_evaluation


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation entry point.
    """
    parser = argparse.ArgumentParser(
        description="Run the BioScanCast evaluation pipeline."
    )
    parser.add_argument(
        "--forecasts",
        default=str(BASE_DIR / "bioscancast_forecasts.csv"),
        help="Path to the forecasts CSV file.",
    )
    parser.add_argument(
        "--questions",
        default=str(BASE_DIR / "bioscancast_questions.csv"),
        help="Path to the questions CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for running the evaluation pipeline from the command line.
    """
    args = parse_args()
    run_evaluation(
        forecasts_path=args.forecasts,
        questions_path=args.questions,
    )


if __name__ == "__main__":
    main()