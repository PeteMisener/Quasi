"""
SmartPeriodIndex: Normalizes temporal cadences, quarters, dates, and integers
into contiguous zero-indexed integers for panel econometrics without warnings.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class SmartPeriodIndex:
    """
    Normalizes diverse time formats (quarters like '2023Q1', monthly dates,
    ISO timestamps, or raw integers) into a contiguous 0-indexed integer sequence.
    """

    def __init__(self, raw_uniques: Sequence[Any]) -> None:
        self.raw_uniques = list(raw_uniques)
        self.labels: List[str] = []
        self.time_to_idx: Dict[Any, int] = {}
        self.idx_to_time: Dict[int, Any] = {}
        self._cadence: str = "unknown"
        self._build_index()
        self._cadence = self._infer_cadence()

    def _build_index(self) -> None:
        raw = self.raw_uniques

        # Handle numeric/integer sequences directly.
        if all(isinstance(x, (int, np.integer)) for x in raw):
            sorted_raw = sorted(raw)
        else:
            # Parse dates/quarters suppressing pandas element-wise format warnings.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    parsed = pd.to_datetime(raw)
                sorted_raw = [x for _, x in sorted(zip(parsed, raw))]
            except Exception:
                # Fallback to string sort if date parsing fails entirely.
                sorted_raw = sorted(raw, key=lambda x: str(x))

        self.labels = [str(x) for x in sorted_raw]

        for idx, val in enumerate(sorted_raw):
            self.time_to_idx[val] = idx
            self.idx_to_time[idx] = val

    @property
    def cadence(self) -> str:
        """Return a human-readable temporal cadence for model metadata."""
        return self._cadence

    def _infer_cadence(self) -> str:
        """Infer the temporal cadence represented by the raw time labels."""
        raw = list(self.raw_uniques)

        if len(raw) < 2:
            return "unknown"

        # Integer periods are common in panel data. Four-digit consecutive
        # years are annual; other integer sequences are generic numeric periods.
        if all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in raw):
            ordered = sorted(int(x) for x in raw)
            diffs = np.diff(ordered)

            if (
                len(ordered) >= 2
                and np.all(diffs == 1)
                and all(1000 <= x <= 9999 for x in ordered)
            ):
                return "annual"

            return "numeric"

        # Preserve explicit quarter semantics.
        quarter_pattern = re.compile(r"^\d{4}[- ]?Q[1-4]$", re.IGNORECASE)

        if all(isinstance(x, str) and quarter_pattern.fullmatch(x) for x in raw):
            return "quarterly"

        try:
            parsed = pd.to_datetime(raw)
        except (TypeError, ValueError):
            return "irregular"

        try:
            freq = pd.infer_freq(pd.DatetimeIndex(sorted(parsed)))
        except (TypeError, ValueError):
            freq = None

        if freq:
            freq_upper = str(freq).upper()

            if freq_upper.startswith(("YS", "YE", "Y")):
                return "annual"

            if freq_upper.startswith(("QS", "QE", "Q")):
                return "quarterly"

            if freq_upper.startswith(("MS", "ME", "M")):
                return "monthly"

            if freq_upper.startswith("W"):
                return "weekly"

            if freq_upper.startswith(("D", "B")):
                return "daily"

            if freq_upper.startswith(("H", "T", "MIN", "S")):
                return "subdaily"

        return "irregular"

    def transform(self, series: Union[pd.Series, Sequence[Any]]) -> np.ndarray:
        """Transform an input time series into normalized contiguous integers."""
        if isinstance(series, pd.Series):
            return series.map(self.time_to_idx).to_numpy(dtype=int)

        return np.array([self.time_to_idx[x] for x in series], dtype=int)

    def inverse_transform(self, indices: Sequence[int]) -> List[Any]:
        """Map contiguous integer indices back to their original representations."""
        return [self.idx_to_time[i] for i in indices]

    @classmethod
    def from_series(
        cls, series: Union[pd.Series, Sequence[Any]]
    ) -> Tuple["SmartPeriodIndex", np.ndarray]:
        """
        Convenience constructor: builds the index and transforms the series.

        Returns:
            Tuple of (SmartPeriodIndex instance, integer index array)
        """
        if isinstance(series, pd.Series):
            uniques = series.dropna().unique()
        else:
            uniques = np.unique(series)

        indexer = cls(uniques)
        integers = indexer.transform(series)

        return indexer, integers