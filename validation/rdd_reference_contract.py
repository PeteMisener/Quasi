"""Shared, fail-closed RDD parity conventions.

This validates data *integrity and compatibility*, not whether a third-party
reference was truly executed. A signed CI artifact and external execution
history are still needed for independent parity evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.rdd_fixture_contract import PHASE7_RDD_SHA256

PARITY_DESIGNS = ("sharp", "fuzzy")
PARITY_METHODS = ("conventional", "bias_corrected")
PARITY_H = 1.15
PARITY_B = 1.35
PARITY_P = 1
PARITY_Q = 2
PARITY_VCE = "hc0"
PARITY_KERNEL = "tri"
PARITY_CUTOFF = 0.0

# Exact values are required rather than broad comparisons on run parameters.
REFERENCE_FIELDS = (
    "design", "method", "estimate", "se", "reference_package",
    "reference_version", "fixture_sha256", "h", "b", "p", "q",
    "vce", "kernel", "cutoff", "generator",
)


def validate_r_reference(frame: pd.DataFrame, fixture_hash: str) -> pd.DataFrame:
    """Reject stale, malformed or incompatible independently generated R rows."""
    if list(frame.columns) != list(REFERENCE_FIELDS):
        raise ValueError("R reference schema/options provenance mismatch")
    if fixture_hash != PHASE7_RDD_SHA256:
        raise ValueError("Invalid fixture fingerprint supplied to R reference validator")
    expected = {(design, method) for design in PARITY_DESIGNS for method in PARITY_METHODS}
    actual = list(zip(frame["design"], frame["method"]))
    if len(frame) != len(expected) or set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("R reference requires exactly one row for each sharp/fuzzy and inference method")
    for col, literal in (
        ("fixture_sha256", fixture_hash), ("reference_package", "rdrobust"),
        ("generator", "validation/r/phase7_rd_external_reference.R"),
        ("vce", PARITY_VCE), ("kernel", PARITY_KERNEL),
    ):
        if frame[col].isna().any() or not frame[col].eq(literal).all():
            raise ValueError(f"R reference provenance mismatch for {col}")
    if frame["reference_version"].isna().any() or any(
        not str(v).strip() or str(v).strip().lower() in ("unknown", "nan", "test_double_not_authentic")
        for v in frame["reference_version"]
    ) or frame["reference_version"].nunique() != 1:
        raise ValueError("R reference package version missing, inconsistent, or not credible")
    for col, expected_value in (
        ("h", PARITY_H), ("b", PARITY_B),
        ("p", PARITY_P), ("q", PARITY_Q), ("cutoff", PARITY_CUTOFF),
    ):
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all() or not np.allclose(vals, expected_value, rtol=0, atol=1e-12):
            raise ValueError(f"R reference settings mismatch for {col}")
    for col in ("estimate", "se"):
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).all() or (col == "se" and (vals <= 0).any()):
            raise ValueError(f"R reference has invalid {col}")
    return frame
