# BioScanCast Evaluation Stage

This folder evaluates probabilistic forecasts and tracks how they change across repeated forecast versions.

## Purpose

The eval stage is designed for forecasting systems that improve over time. In the demo data, `forecast_version` is the time axis:

- `1` = earliest forecast
- `2` = updated forecast
- `3` = latest forecast

The plots and tables are built to show whether a source is improving as versions advance.

## What it measures

The evaluation stage scores each question/source/version trio using metrics where lower is better:

- **Brier score**: probability error.
- **Log score**: negative log probability assigned to the true answer.
- **Accuracy error**: `1 - accuracy` so the direction matches the other metrics.
- **RPS**: Ranked Probability Score for ordered buckets.
- **Top probability**: the largest probability in the forecast.
- **Normalized entropy**: forecast uncertainty.
- **True probability**: probability assigned to the actual resolved outcome.

## Input format

Forecast files must contain at least:

- `question_id`
- `forecast_source`
- `forecast_version`
- `option`
- `probability`

`forecast_version` is the versioned update number, not a random run id. Use it to place repeated forecasts on the timeline.

Example:

```csv
question_id;forecast_source;forecast_version;option;probability
q1;bioscancast;1;70-100;0.70
q1;bioscancast;2;70-100;0.45
q1;bioscancast;3;70-100;0.15
```

## How to run

From the project root:

```bash
python -m bioscancast.stages.eval_stage.main
```

To run custom forecasts, pass one or more CSV files:

```bash
python -m bioscancast.stages.eval_stage.main --forecasts bioscancast/stages/eval_stage/mock_forecasts/human_forecasts.csv bioscancast/stages/eval_stage/mock_forecasts/bioscancast_forecasts.csv bioscancast/stages/eval_stage/mock_forecasts/llm_baseline_forecasts.csv
```

## Outputs

The pipeline writes results to `bioscancast/stages/eval_stage/outputs/`.

Key files:

- `question_level_metrics.csv`
- `summary_metrics.csv`
- `summary_metrics_by_question_type.csv`
- `summary_metrics_over_time.csv`
- `source_ranking_over_time.csv`
- `metric_improvement_over_time.csv`
- `score_timeline_boxplots.png`
- `source_timeline_summary.png`
- `improvement_vs_v1.png`
- `question_heatmap.png`
- `source_ranking_over_time.png`

### Main plots

- `score_timeline_boxplots.png` shows the distribution of question scores for each source and version.
- `source_timeline_summary.png` shows the median trajectory with interquartile-range bands.
- `improvement_vs_v1.png` shows how much each version improves over version 1, where positive values mean better performance.
- `question_heatmap.png` shows question-level evolution across versions.
- `source_ranking_over_time.png` shows the source ranking by median Brier score at each version.

## Perplexity forecasts

To generate a new CSV from Perplexity, use the forecast generator and point it at the question file and a template forecast file so it can reuse the same option sets:

```bash
python -m bioscancast.stages.eval_stage.generate_perplexity_forecasts --questions bioscancast/stages/eval_stage/bioscancast_questions.csv --template-forecasts bioscancast/stages/eval_stage/mock_forecasts/bioscancast_forecasts.csv --output bioscancast/stages/eval_stage/mock_forecasts/perplexity_forecasts.csv
```

The script expects `PERPLEXITY_API_KEY` to be set in the environment.

## Interpretation

If BioScanCast learns over time, we expect:

- Brier score down
- Log score down
- RPS down
- Accuracy error down

from version 1 to version 3.

The versioned demo mocks in this folder are set up to make those trends visible.
