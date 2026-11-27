#!/usr/bin/env python3
"""
Standalone runner for quasi Phase 8 R-Parity Benchmarks.
Executes assertions and prints a scannable comparison table in the console.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rich.console import Console
from rich.table import Table

import quasi
from tests.fixtures.seed_data import (
    ensure_benchmark_files_exist,
    get_mpdta_benchmark,
    get_prop99_benchmark,
    get_senate_benchmark,
)


def run_benchmark_suite() -> int:
    """Deprecated legacy parity-reporting entry point (deliberately fail closed).

    The previous implementation compared synthetic fallback data against embedded
    reference-like numbers and incorrectly labeled matches as independent R parity.
    Until the Phase 3 reference-laboratory work is complete, use the synthetic
    smoke tests without presenting their output as canonical package agreement.
    """
    raise RuntimeError(
        "The legacy 'R parity matrix' is disabled: authentic datasets, "
        "pinned reference outputs, and method-compatible options are missing. "
        "Run `PYTHONPATH=src:. python -m pytest tests/synthetic_regression` "
        "for synthetic compatibility smoke tests; these are NOT R parity tests."
    )


if __name__ == "__main__":
    sys.exit(run_benchmark_suite())
