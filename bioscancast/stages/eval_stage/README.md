# BioScanCast Evaluation Stage

This module evaluates probabilistic forecasts using:

- Multiclass Brier score
- Log score
- Accuracy
- Calibration summaries
- Simple visualisations

The pipeline compares forecast sources such as:

- human/community forecasts
- LLM forecasts
- ensemble forecasts

using resolved real-world outcomes.

---

# Folder Structure

```text
bioscancast/
└── stages/
    └── eval_stage/
        ├── __init__.py
        ├── calibration.py
        ├── compare.py
        ├── loaders.py
        ├── main.py
        ├── pipeline.py
        ├── schemas.py
        ├── scoring.py
        ├── visualisations.py
        ├── bioscancast_forecasts.csv
        ├── bioscancast_questions.csv
        └── outputs/