"""Interrupted Time Series module for quasi."""

from __future__ import annotations

from quasi.its.controlled import CITSResult, ControlledITS
from quasi.its.segmented import (
    ITSResult,
    InterruptedTimeSeries,
    compute_newey_west_hac,
)

__all__ = [
    "InterruptedTimeSeries",
    "ITSResult",
    "ControlledITS",
    "CITSResult",
    "compute_newey_west_hac",
]
