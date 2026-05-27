"""Thin wrapper around ``python -m bioscancast.main`` so the orchestrator
matches the existing per-stage runner convention (scripts/run_*.py).

Usage:
    python scripts/run_pipeline.py q7 --as-of-date 2025-02-28 -v
"""

from __future__ import annotations

import os
import sys

# Add project root to path so `bioscancast` imports work when run from
# anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bioscancast.main import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
