from __future__ import annotations
 
import argparse
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
 
import pandas as pd
from openai import OpenAI
 
from bioscancast.stages.evaluation.loaders import load_forecasts, load_questions
 
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = BASE_DIR / "bioscancast_questions_resolved.csv"
DEFAULT_TEMPLATE_FORECASTS = BASE_DIR / "mock_forecasts" / "llm_baseline_forecasts.csv"
DEFAULT_OUTPUT = BASE_DIR / "mock_forecasts" / "perplexity_forecasts.csv"
 
MODEL_CHOICES = ("sonar-pro", "sonar-reasoning-pro", "sonar-deep-research")
 
# Columns/format used both for reading back an existing perplexity_forecasts.csv
# and for writing the new one. Kept in one place so read/write always agree.
CSV_KWARGS = dict(sep=";", decimal=",", encoding="cp1252")
OUTPUT_COLUMNS = ["question_id", "forecast_source", "forecast_version", "option", "probability", "forecast_date"]
 
 
def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
 
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
 
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"Could not locate a JSON object in the response: {text[:500]}")
 
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from Perplexity.")
    return parsed
 
 
def _options_from_template(template_group: pd.DataFrame) -> List[str]:
    options: List[str] = []
    seen = set()
    for raw in template_group["option"].tolist():
        option = str(raw).strip()
        if option and option not in seen:
            options.append(option)
            seen.add(option)
    if not options:
        raise ValueError("Template forecast group has no options.")
    return options
 
 
def _build_prompt(question_row: pd.Series, options: List[str]) -> str:
    options_block = "\n".join(f"- {option}" for option in options)
    return f"""You are writing a probabilistic forecast for a benchmark dataset.
 
Question ID: {question_row['question_id']}
Topic: {question_row.get('topic', '')}
Question text: {question_row.get('question_text', '')}
Question type: {question_row.get('question_type', '')}
Resolution criteria: {question_row.get('resolution_criteria', '')}
 
Choose probabilities for exactly these options:
{options_block}
 
Return ONLY valid JSON with this exact schema:
{{
  "probabilities": {{
    "option text 1": 0.0,
    "option text 2": 0.0
  }}
}}
 
Rules:
- include every option exactly once
- probabilities must be numbers between 0 and 1
- probabilities should sum to 1.0
- do not include markdown, explanations, or code fences
""".strip()
 
 
def _parse_probability_map(payload: Dict[str, Any], options: List[str]) -> Dict[str, float]:
    if isinstance(payload.get("probabilities"), dict):
        raw_map = payload["probabilities"]
    elif all(option in payload for option in options):
        raw_map = payload
    elif isinstance(payload.get("options"), list):
        raw_map = {}
        for item in payload["options"]:
            if isinstance(item, dict) and "option" in item and "probability" in item:
                raw_map[str(item["option"]).strip()] = item["probability"]
    else:
        raise ValueError("Could not find a probability mapping in the response JSON.")
 
    result: Dict[str, float] = {}
    for option in options:
        value = raw_map.get(option, 0.0)
        try:
            result[option] = max(float(value), 0.0)
        except Exception as exc:
            raise ValueError(f"Invalid probability for option {option!r}: {value!r}") from exc
 
    total = sum(result.values())
    if total <= 0:
        uniform = 1.0 / len(options)
        return {option: uniform for option in options}
 
    return {option: value / total for option, value in result.items()}
 
 
def _get_client() -> OpenAI:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY is not set.")
    return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
 
 
def _generate_rows_for_question(
    client: OpenAI,
    model: str,
    question_row: pd.Series,
    options: List[str],
    version: int,
    forecast_date: str,
) -> List[Dict[str, Any]]:
    prompt = _build_prompt(question_row, options)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
 
    content = response.choices[0].message.content or ""
    parsed = _extract_json_object(content)
    probability_map = _parse_probability_map(parsed, options)
 
    return [
        {
            "question_id": question_row["question_id"],
            "forecast_source": "perplexity",
            "forecast_version": version,
            "option": option,
            "probability": probability_map[option],
            "forecast_date": forecast_date,
        }
        for option in options
    ]
 
 
def _next_version(output_path: str | Path) -> int:
    """Look at whatever is already in output_path and return last_version + 1.
 
    Returns 1 if the file doesn't exist yet or has no forecast_version values.
    """
    path = Path(output_path)
    if not path.exists():
        return 1
 
    try:
        existing = pd.read_csv(path, **CSV_KWARGS)
    except Exception:
        # Empty or unreadable file -> treat as no history.
        return 1
 
    if "forecast_version" not in existing.columns or existing.empty:
        return 1
 
    try:
        return int(pd.to_numeric(existing["forecast_version"], errors="coerce").max()) + 1
    except (ValueError, TypeError):
        return 1
 
 
def _load_existing(output_path: str | Path) -> Optional[pd.DataFrame]:
    path = Path(output_path)
    if not path.exists():
        return None
    try:
        existing = pd.read_csv(path, **CSV_KWARGS)
    except Exception:
        return None
    if existing.empty:
        return None
    return existing
 
 
def build_forecasts(
    questions_path: str | Path,
    template_forecasts_path: str | Path,
    model: str,
    version: int,
    forecast_date: str,
) -> List[Dict[str, Any]]:
    questions_df = load_questions(questions_path)
    template_df = load_forecasts(template_forecasts_path)
 
    if "question_id" not in questions_df.columns:
        raise ValueError("Questions file must contain question_id.")
    if "question_id" not in template_df.columns or "option" not in template_df.columns:
        raise ValueError("Template forecasts must contain question_id and option columns.")
 
    questions_df = questions_df.copy()
    questions_df["question_id"] = questions_df["question_id"].astype(str).str.strip()
    question_lookup = questions_df.set_index("question_id", drop=False)
 
    client = _get_client()
    rows: List[Dict[str, Any]] = []
 
    for question_id, group in template_df.groupby("question_id", sort=False):
        question_id = str(question_id).strip()
        if question_id not in question_lookup.index:
            raise KeyError(
                f"Question {question_id!r} is present in the template forecasts but missing from {questions_path}."
            )
 
        question_row = question_lookup.loc[question_id]
        options = _options_from_template(group)
        rows.extend(
            _generate_rows_for_question(client, model, question_row, options, version, forecast_date)
        )
 
    return rows
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Perplexity forecast CSV for BioScanCast.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="Path to bioscancast_questions.csv.")
    parser.add_argument(
        "--template-forecasts",
        default=str(DEFAULT_TEMPLATE_FORECASTS),
        help="Forecast CSV used to reuse the option sets.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write/append the generated forecast CSV.")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="sonar-reasoning-pro",
        help="Perplexity model to query.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing output file: start versioning at 1 and replace the file instead of appending.",
    )
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
 
    version = 1 if args.overwrite else _next_version(output_path)
    forecast_date = date.today().isoformat()
 
    rows = build_forecasts(args.questions, args.template_forecasts, args.model, version, forecast_date)
    new_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
 
    existing_df = None if args.overwrite else _load_existing(output_path)
    if existing_df is not None:
        # Align columns (older files may predate forecast_date) and stack old + new.
        for col in OUTPUT_COLUMNS:
            if col not in existing_df.columns:
                existing_df[col] = ""
        existing_df = existing_df[OUTPUT_COLUMNS]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False, **CSV_KWARGS)
    print(f"Wrote {len(new_df)} new rows (version {version}, {forecast_date}); file now has {len(combined_df)} rows total: {output_path}")
 
 
if __name__ == "__main__":
    main()