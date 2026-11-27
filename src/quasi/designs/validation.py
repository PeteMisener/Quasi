"""Fail-closed data contract for explicit Quasi design specifications.

No cleaning, imputation, sorting or row drops occur in this module. Return an
immutable audit so downstream consumers can document the observed sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np
import pandas as pd

from quasi.designs.specs import (
    DesignSpec, PanelDesign, RDDDesign, IVDesign, ITSDesign, SyntheticDesign, ObservationalDesign,
)
from quasi.errors import InvalidDesignError, InsufficientSupportError, NotIdentifiedError


@dataclass(frozen=True)
class DesignAudit:
    family: str
    n_rows: int
    n_units: int | None
    n_periods: int | None
    n_excluded: int = 0
    warnings: Tuple[str, ...] = ()
    checks: Tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _as_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "to_pandas"):
        converted = data.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise InvalidDesignError("Data must be a pandas DataFrame or convertible with to_pandas()")


def _regular_time(series: pd.Series) -> bool:
    """Check contiguous numeric or identifiable calendar cadence, not just sorted order."""
    ordered = pd.Index(series.drop_duplicates().sort_values())
    if len(ordered) < 3:
        return False
    if pd.api.types.is_numeric_dtype(ordered.dtype):
        vals = ordered.to_numpy(dtype=float)
        return bool(np.allclose(np.diff(vals), np.diff(vals)[0], atol=1e-10, rtol=1e-9))
    if isinstance(ordered, pd.PeriodIndex):
        return bool(np.all(np.diff(ordered.asi8) == 1))
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dates = pd.DatetimeIndex(pd.to_datetime(ordered, errors="raise"))
        return pd.infer_freq(dates) is not None
    except (TypeError, ValueError):
        return False


def validate_design_data(design: DesignSpec, data: Any) -> DesignAudit:
    """Validate before calling experimental estimators; does not imply identification."""
    if type(design) not in (PanelDesign, RDDDesign, IVDesign, ITSDesign, SyntheticDesign, ObservationalDesign):
        raise InvalidDesignError("Provide a supported explicit DesignSpec, not an arbitrary object")
    df = _as_frame(data)
    if df.empty:
        raise InsufficientSupportError("Dataset is empty")
    roles = [design.outcome, *design.covariates]
    if design.cluster:
        roles.append(design.cluster)
    if isinstance(design, (PanelDesign, SyntheticDesign)):
        roles += [design.unit, design.time]
    if isinstance(design, SyntheticDesign) and design.treatment:
        roles.append(design.treatment)
    if isinstance(design, PanelDesign):
        roles.append(design.treatment or design.first_treated)
    if isinstance(design, RDDDesign):
        roles += [design.running_var, *design.diagnostic_covariates]
        if design.treatment:
            roles.append(design.treatment)
    if isinstance(design, IVDesign):
        roles += [design.endogenous, *design.instruments]
    if isinstance(design, ITSDesign):
        roles.append(design.time)
    if isinstance(design, ObservationalDesign):
        roles.append(design.treatment)
    missing = sorted(set(roles) - set(df.columns))
    if missing:
        raise InvalidDesignError(f"Missing required columns: {missing}")
    if df[roles].isna().any().any():
        bad = df[roles].columns[df[roles].isna().any()].tolist()
        # Never do listwise deletion without an explicit policy.
        # First-treated cohort column is special: NaN encodes never-treated.
        if isinstance(design, PanelDesign) and design.first_treated:
            bad = [x for x in bad if x != design.first_treated]
        if bad:
            raise InvalidDesignError(f"Missing values in {bad}; missing_policy='raise' prohibits silent deletion")
    numeric = [design.outcome, *design.covariates]
    if isinstance(design, RDDDesign):
        numeric.extend([design.running_var, *design.diagnostic_covariates])
        if design.treatment:
            numeric.append(design.treatment)
    if isinstance(design, IVDesign):
        numeric += [design.endogenous, *design.instruments]
    if isinstance(design, ObservationalDesign):
        numeric.append(design.treatment)
    if isinstance(design, PanelDesign) and design.treatment:
        numeric.append(design.treatment)
    if isinstance(design, SyntheticDesign) and design.treatment:
        numeric.append(design.treatment)
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            raise InvalidDesignError(f"Column '{col}' must have a numeric dtype, not numeric-looking strings")
        try:
            values = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as err:
            raise InvalidDesignError(f"Column '{col}' must be numeric and finite") from err
        if not np.isfinite(values).all():
            raise InvalidDesignError(f"Column '{col}' contains infinite or nonfinite values")
    n_units = n_periods = None
    warnings: list[str] = []
    checks: list[str] = ["columns_exist", "missing_data_explicit", "finite_numeric_values"]

    if isinstance(design, (PanelDesign, SyntheticDesign)):
        n_units, n_periods = df[design.unit].nunique(), df[design.time].nunique()
        if df.duplicated([design.unit, design.time]).any():
            raise InvalidDesignError("Duplicate unit-time keys; one observation per unit-period required")
        if len(df) != n_units * n_periods:
            raise InvalidDesignError("Unbalanced panel: some unit-time observations are missing")
        if n_units < 2 or n_periods < 3:
            raise InsufficientSupportError("Panel requires at least two units and three periods")
        checks += ["unique_panel_keys", "balanced_panel"]
        if not _regular_time(df[design.time]):
            warnings.append(
                "Panel period spacing could not be verified; models index observed periods "
                "ordinally, not by physical elapsed time"
            )
        if isinstance(design, PanelDesign):
            if design.treatment:
                if not df[design.treatment].isin([0, 1, False, True]).all():
                    raise InvalidDesignError("Treatment must be binary 0/1")
                paths = df.sort_values(design.time).groupby(design.unit, sort=False)[design.treatment]
                histories = [s.to_numpy(dtype=int) for _, s in paths]
                if any(np.any(np.diff(path) < 0) for path in histories):
                    raise InvalidDesignError("Reversible treatment unsupported: treatment must be absorbing")
                if any(np.argmax(path) <= design.anticipation for path in histories if path[-1] == 1):
                    raise NotIdentifiedError("First observed period or anticipation window has no clean pretreatment baseline")
                if not any(path[-1] == 1 for path in histories):
                    raise InsufficientSupportError("No treated units found")
                if design.control_group == "never_treated" and not any(path[-1] == 0 for path in histories):
                    raise InsufficientSupportError("No never-treated units for requested control group")
            else:
                by_unit = df.groupby(design.unit)[design.first_treated]
                if any(g.nunique(dropna=False) > 1 for _, g in by_unit):
                    raise InvalidDesignError("first_treated must be constant per unit, including never-treated missingness")
                times = list(pd.Index(df[design.time].unique()).sort_values())
                observed = df[design.first_treated].dropna()
                # R did gname convention: zero encodes never-treated when zero
                # is not an observed period; -1 is a legacy equivalent.
                never_sentinels = {x for x in (-1, 0) if x not in times}
                observed = observed[~observed.isin(never_sentinels)]
                if any(t not in times for t in observed):
                    raise InvalidDesignError("first_treated contains adoption times outside observed panel")
                if any(times.index(t) <= design.anticipation for t in observed):
                    raise NotIdentifiedError("First observed period or anticipation window has no clean pretreatment baseline")
                if observed.empty:
                    raise InsufficientSupportError("No adopting units found")
                if design.control_group == "never_treated" and not (
                    df[design.first_treated].isna() | df[design.first_treated].isin(never_sentinels)
                ).any():
                    raise InsufficientSupportError("No never-treated units for requested control group")
            checks.append("absorbing_treatment_support")
        else:
            if not (df[design.unit] == design.treated_unit).any():
                raise InsufficientSupportError("treated_unit is not present in panel")
            times = list(df[design.time].unique())
            if design.intervention_time not in times:
                raise InvalidDesignError("intervention_time must be an observed period")
            sorted_times = sorted(times)
            k = sorted_times.index(design.intervention_time)
            if k < 2 or len(sorted_times) - k < 1:
                raise InsufficientSupportError("Synthetic design needs at least 2 pretreatment periods and a post period")
            if n_units < 3:
                raise InsufficientSupportError("Synthetic design needs at least two donor units")
            if design.treatment:
                # Same treatment histories as the direct constructor must be checked
                # before the estimator is called; this is *not* a spillover test.
                from quasi.synth.panel import prepare_synthetic_panel
                prepare_synthetic_panel(df, outcome=design.outcome, unit=design.unit,
                                        time=design.time, treated_unit=design.treated_unit,
                                        intervention_time=design.intervention_time,
                                        treatment=design.treatment)
                checks.append("observed_synthetic_treatment_history")
            else:
                warnings.append("Donor treatment histories were not supplied; donor eligibility remains unverified")
            checks.append("synthetic_treated_donor_prepost_support")
    elif isinstance(design, RDDDesign):
        x = df[design.running_var].to_numpy(dtype=float)
        if not (np.any(x < design.cutoff) and np.any(x >= design.cutoff)):
            raise InsufficientSupportError("RDD requires observations on both sides of cutoff")
        if isinstance(design.bandwidth, (float, int)) and not isinstance(design.bandwidth, bool):
            mask = (np.abs(x - design.cutoff) <= design.bandwidth) & (
                (np.abs(x - design.cutoff) > design.donut_radius) if design.donut_radius else True)
            if np.sum(mask & (x < design.cutoff)) < design.order + 2 or np.sum(mask & (x >= design.cutoff)) < design.order + 2:
                raise InsufficientSupportError("Insufficient support on both sides within requested bandwidth")
        if design.treatment:
            if not df[design.treatment].isin([0, 1, False, True]).all():
                raise InvalidDesignError("Fuzzy RDD treatment must be binary")
            if df[design.treatment].nunique() < 2:
                raise InsufficientSupportError("Fuzzy RDD requires observed treatment variation")
            warnings.append("Fuzzy RDD first-stage discontinuity at the cutoff is not established by preflight")
        if design.donut_radius:
            warnings.append("Donut RD extrapolates through an excluded region; exclusion is not a manipulation test")
        checks.append("cutoff_two_sided_support")
    elif isinstance(design, ITSDesign):
        if df[design.time].duplicated().any():
            raise InvalidDesignError("ITS time column must have unique observations")
        times = df[design.time]
        ordered = list(pd.Index(times).sort_values())
        if design.intervention_date not in ordered:
            raise InvalidDesignError("intervention_date must occur in observed time column")
        k = ordered.index(design.intervention_date)
        if k < 3 or len(ordered) - k < 3:
            raise InsufficientSupportError("ITS requires 3 pre- and 3 post-intervention observations")
        if not _regular_time(times):
            raise InvalidDesignError("ITS requires a regular time cadence; gaps/unequal steps are unsupported")
        checks += ["unique_time_points", "regular_time_cadence", "prepost_support"]
    elif isinstance(design, IVDesign):
        # Matrix rank checks in phase 5 will add tolerances and instrument relevance.
        if len(df) < len(design.covariates) + len(design.instruments) + 3:
            raise InsufficientSupportError("IV sample has too few observations for its design matrix")
        for col in [design.endogenous, *design.instruments]:
            if df[col].nunique() < 2:
                raise InsufficientSupportError(f"No variation in IV column '{col}'")
        checks.append("instrument_presence_and_variation")
    elif isinstance(design, ObservationalDesign):
        if not df[design.treatment].isin([0, 1, False, True]).all():
            raise InvalidDesignError("Observational treatment must be binary")
        if df[design.treatment].nunique() != 2:
            raise InsufficientSupportError("Need both treated and comparison units")
        warnings.append("Observational overlap and conditional exchangeability are not established")
        checks.append("binary_treatment_support")
    return DesignAudit(design.family, len(df), n_units, n_periods, warnings=tuple(warnings), checks=tuple(checks))
