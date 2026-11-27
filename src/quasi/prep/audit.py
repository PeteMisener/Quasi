"""Automated pre-flight panel health audit engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal
import numpy as np
import pandas as pd


@dataclass
class PanelHealthAudit:
    rectangularity_ratio: float
    is_balanced: bool
    treatment_type: Literal["absorbing_staggered", "reversible", "single_shot", "static"]
    missing_period_count: int
    small_cluster_flag: bool
    cluster_count: int
    is_viable: bool
    blocker_reasons: List[str]
    warnings: List[str]
    duplicate_key_count: int = 0

    def format_banner(self) -> str:
        lines = []
        if self.duplicate_key_count > 0:
            lines.append(
                f"Duplicate unit-time keys detected ({self.duplicate_key_count} rows). "
                "Resolve duplicates before panel estimation."
            )
        if not self.is_balanced:
            lines.append(
                f"Panel is imbalanced ({self.rectangularity_ratio * 100:.1f}% complete). "
                "Balanced-panel estimators reject missing unit-periods rather than dropping rows silently."
            )
        if self.small_cluster_flag:
            lines.append(
                f"Small cluster count (G = {self.cluster_count} < 30). "
                "Asymptotic cluster inference may under-cover; consider CR2/wild cluster bootstrap."
            )
        if self.treatment_type == "reversible":
            lines.append(
                "Reversible treatment path detected. Standard Callaway-Sant'Anna requires absorbing treatment."
            )
        return "\n".join(f"[WARN] {line}" for line in lines)


def audit_panel(
    data: pd.DataFrame,
    unit: str,
    time: str,
    treatment: str,
) -> PanelHealthAudit:
    """Executes structural and mathematical audit on panel datasets."""
    blockers: List[str] = []
    warnings: List[str] = []

    missing_cols = [c for c in (unit, time, treatment) if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for panel audit: {missing_cols}")

    if data[[unit, time, treatment]].isna().any().any():
        blockers.append("Missing values present in unit, time, or treatment columns.")

    dup_count = int(data.duplicated([unit, time]).sum())
    if dup_count > 0:
        blockers.append(f"Found {dup_count} duplicate unit-time keys.")

    valid_keys = data.dropna(subset=[unit, time])
    units = valid_keys[unit].unique()
    times = valid_keys[time].unique()
    expected_rows = len(units) * len(times)
    actual_rows = len(valid_keys.drop_duplicates([unit, time]))
    rect_ratio = actual_rows / expected_rows if expected_rows > 0 else 0.0

    if rect_ratio < 0.999:
        warnings.append(f"Panel incomplete ({actual_rows}/{expected_rows} unique unit-periods observed).")

    tx_nonnull = data[treatment].dropna()
    is_binary_tx = bool(tx_nonnull.isin([0, 1, False, True]).all())
    if not is_binary_tx:
        blockers.append("Treatment column contains non-binary values outside {0, 1}.")

    # Check rollout dynamics using actual time labels (not positional offsets)
    switches_off = False
    first_adopt_times = set()
    if is_binary_tx and len(tx_nonnull) > 0:
        clean_df = data.dropna(subset=[unit, time, treatment]).sort_values([unit, time])
        for _, grp in clean_df.groupby(unit, sort=False):
            arr = grp[treatment].to_numpy(dtype=float)
            t_vals = grp[time].to_numpy()
            if len(arr) > 1 and np.any(np.diff(arr) < 0):
                switches_off = True
            if np.any(arr == 1):
                first_adopt_times.add(t_vals[int(np.argmax(arr == 1))])

    if switches_off:
        treatment_type: Literal["absorbing_staggered", "reversible", "single_shot", "static"] = "reversible"
    elif not first_adopt_times:
        treatment_type = "static"
    elif len(first_adopt_times) == 1:
        treatment_type = "single_shot"
    else:
        treatment_type = "absorbing_staggered"

    n_clusters = len(units)
    small_cluster = n_clusters < 30
    if small_cluster:
        warnings.append(f"Cluster count G = {n_clusters} is less than 30.")

    if is_binary_tx:
        treat_sum = float(tx_nonnull.astype(float).sum()) if len(tx_nonnull) else 0.0
        if treat_sum == 0:
            blockers.append("Zero treated observations found. Identification is mathematically impossible.")
        elif treat_sum == len(tx_nonnull) and len(tx_nonnull) > 0:
            blockers.append("Zero comparison/control observations found. All rows are treated.")

    return PanelHealthAudit(
        rectangularity_ratio=rect_ratio,
        is_balanced=(dup_count == 0 and rect_ratio >= 0.999999),
        treatment_type=treatment_type,
        missing_period_count=max(0, expected_rows - actual_rows),
        small_cluster_flag=small_cluster,
        cluster_count=n_clusters,
        is_viable=(len(blockers) == 0),
        blocker_reasons=blockers,
        warnings=warnings,
        duplicate_key_count=dup_count,
    )
