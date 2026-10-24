#!/usr/bin/env python
"""Thin CLI wrapper — see `python -m tribunal.rescue --help` for the same
functionality. Runs the partiality / special-obligations rescue study.

Examples:
    python scripts/run_rescue.py --family claude --smoke-test
    python scripts/run_rescue.py --family claude --trials 10
    python scripts/run_rescue.py --family both --trials 10        # needs GEMINI_API_KEY
    python scripts/run_rescue.py --aggregate-only                 # recompute summary/figure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from tribunal.rescue import main  # noqa: E402

if __name__ == "__main__":
    main()
