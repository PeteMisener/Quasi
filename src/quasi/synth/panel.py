"""Strict, shared balanced-panel preparation for experimental synthetic designs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quasi.errors import InvalidDesignError, InsufficientSupportError
from quasi.prep.period_index import SmartPeriodIndex


@dataclass(frozen=True)
class SyntheticPanel:
    treated: np.ndarray
    donors: np.ndarray
    donor_ids: tuple[str, ...]
    time_labels: tuple[str, ...]
    intervention_idx: int
    n_rows: int
    treatment_audit: dict[str, Any]


def prepare_synthetic_panel(data: Any, *, outcome: str, unit: str, time: str,
                            treated_unit: str, intervention_time: Any,
                            treatment: str | None = None) -> SyntheticPanel:
    df = data if isinstance(data, pd.DataFrame) else data.to_pandas()
    if treatment is not None and (not isinstance(treatment, str) or not treatment.strip()
                                  or treatment in {outcome, unit, time}):
        raise InvalidDesignError("treatment must name a distinct nonempty treatment indicator column")
    required = {outcome, unit, time} | ({treatment} if treatment else set())
    if not isinstance(df, pd.DataFrame) or not required.issubset(df):
        raise InvalidDesignError(f"Synthetic design missing required columns: {sorted(required - set(df.columns))}")
    if df.empty or df[[outcome, unit, time]].isna().any().any():
        raise InvalidDesignError("Synthetic design requires complete, nonmissing panel values")
    if not pd.api.types.is_numeric_dtype(df[outcome]) or not np.isfinite(df[outcome].to_numpy(dtype=float)).all():
        raise InvalidDesignError("Synthetic outcomes must be finite numeric values")
    if df.duplicated([unit, time]).any():
        raise InvalidDesignError("Duplicate unit-time keys cannot enter synthetic designs")
    if not isinstance(treated_unit, str) or not treated_unit:
        raise InvalidDesignError("treated_unit must be a nonempty string")
    # No silent string conversion of unit ids: an integer 1 is not the string '1'.
    if any(not isinstance(x, str) or not x for x in df[unit].unique()):
        raise InvalidDesignError("Synthetic unit identifiers must be nonempty strings")
    periods, integers = SmartPeriodIndex.from_series(df[time])
    if intervention_time not in periods.time_to_idx:
        raise InvalidDesignError("Intervention time must be present in the panel")
    idx = periods.time_to_idx[intervention_time]
    if idx < 2 or idx >= len(periods.labels):
        raise InsufficientSupportError("Synthetic design requires two pretreatment and one posttreatment periods")
    work = df[[unit, outcome]].copy()
    work["_quasi_period_internal"] = integers
    pivot = work.pivot(index=unit, columns="_quasi_period_internal", values=outcome)
    if pivot.isna().any().any() or pivot.shape[1] != len(periods.labels):
        raise InvalidDesignError("Synthetic panel must be complete and balanced")
    treatment_audit = {
        "status": "not_observed" if treatment is None else "observed_pattern_checked",
        "treatment_column": treatment,
        "donor_treatment_exposure_checked": treatment is not None,
        "spillovers_verified": False,
        "reason": ("No treatment history supplied; donor exposure is an unverified assumption"
                   if treatment is None else "Observed 0/1 histories validated; spillovers still unverified"),
    }
    if treatment is not None:
        values = df[treatment]
        if not (pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)) \
                or not values.isin([0, 1, False, True]).all():
            raise InvalidDesignError("Synthetic treatment history must be binary numeric/boolean 0/1")
        tx = df[[unit, time, treatment]].copy()
        tx["_quasi_period_internal"] = integers
        tx = tx.pivot(index=unit, columns="_quasi_period_internal", values=treatment)
        tx = tx.reindex(index=pivot.index, columns=pivot.columns)
        if tx.isna().any().any():
            raise InvalidDesignError("Treatment histories must be complete for every unit-period")
        if treated_unit not in tx.index:
            raise InsufficientSupportError("Treated unit absent from treatment history")
        observed_treated = tx.loc[treated_unit].to_numpy(dtype=int)
        expected = np.r_[np.zeros(idx, dtype=int), np.ones(len(periods.labels)-idx, dtype=int)]
        if not np.array_equal(observed_treated, expected):
            raise InvalidDesignError("Treated unit treatment history must be 0 before onset and 1 from onset onward")
        donor_history = tx.drop(index=treated_unit).to_numpy(dtype=int)
        if np.any(donor_history):
            raise InvalidDesignError("Synthetic donor units must be untreated (0) throughout the observed panel")
    if treated_unit not in pivot.index:
        raise InsufficientSupportError("Treated unit absent from synthetic panel")
    donors = tuple(str(x) for x in pivot.index if x != treated_unit)
    if len(donors) < 2:
        raise InsufficientSupportError("At least two untreated donor units are required")
    return SyntheticPanel(
        treated=pivot.loc[treated_unit].to_numpy(dtype=float),
        donors=pivot.loc[list(donors)].to_numpy(dtype=float),
        donor_ids=donors, time_labels=tuple(periods.labels),
        intervention_idx=idx, n_rows=len(df), treatment_audit=treatment_audit,
    )
