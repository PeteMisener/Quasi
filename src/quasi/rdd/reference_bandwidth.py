"""Optional, explicitly delegated RDD bandwidth selection via official rdrobust.

This module does not implement a new MSE/CER selector. It wraps the reference
Python rdbwselect output without changing its algorithm. It is intentionally
not a parity certification of Quasi estimation or robust inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd


REFERENCE_SELECTORS = {"rdrobust_mserd": "mserd", "rdrobust_cerrd": "cerrd"}


@dataclass(frozen=True)
class ReferenceBandwidth:
    h: float
    b: float
    selector: str
    reference_version: str
    masspoints: str = "adjust"
    variance: str = "hc0"


def select_reference_bandwidth(
    y: np.ndarray, x: np.ndarray, cutoff: float, kernel: str,
    order: int, selector: str, *, treatment: np.ndarray | None = None,
) -> ReferenceBandwidth:
    """Request common MSE/CER h/b from rdrobust.rdbwselect, without fallback.

    Different left/right bandwidths are unsupported by Quasi's symmetric-window
    inference implementation; we only expose common-bandwidth selectors here.
    """
    if selector not in REFERENCE_SELECTORS:
        raise ValueError(f"Unsupported rdrobust selector: {selector!r}")
    try:
        reference = import_module("rdrobust")
    except ImportError as exc:
        raise ImportError(
            "Bandwidth selection requires the official rdrobust Python package; "
            "install it with 'pip install rdrobust'. No heuristic fallback is used."
        ) from exc
    if not callable(getattr(reference, "rdbwselect", None)):
        raise RuntimeError("Installed rdrobust does not expose rdbwselect; reference selection blocked")
    params = dict(y=np.asarray(y, dtype=float), x=np.asarray(x, dtype=float),
                  c=float(cutoff), p=int(order), q=int(order)+1,
                  kernel={"triangular": "tri", "epanechnikov": "epa", "uniform": "uni"}[kernel],
                  bwselect=REFERENCE_SELECTORS[selector], vce="hc0", masspoints="adjust")
    if treatment is not None:
        params["fuzzy"] = np.asarray(treatment, dtype=float)
    # Never swallow errors from rdrobust (e.g., support/mass-point issues).
    selection = reference.rdbwselect(**params)
    bws = getattr(selection, "bws", None)
    if not isinstance(bws, pd.DataFrame) or bws.shape != (1, 4):
        raise ValueError("rdrobust returned an unexpected bandwidth layout; inspect the installed reference version")
    expected_columns = ["h (left)", "h (right)", "b (left)", "b (right)"]
    if list(bws.columns) != expected_columns or str(bws.index[0]).lower() != REFERENCE_SELECTORS[selector]:
        raise ValueError("rdrobust returned mismatched selector or bandwidth columns")
    vals = bws.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or np.any(vals <= 0):
        raise ValueError("rdrobust returned non-finite or nonpositive bandwidths")
    if not np.isclose(vals[0], vals[1], rtol=1e-9, atol=1e-12) or not np.isclose(vals[2], vals[3], rtol=1e-9, atol=1e-12):
        raise ValueError("Quasi currently supports only common left/right reference bandwidths")
    try:
        reference_version = version("rdrobust")
    except PackageNotFoundError:
        # Makes mock implementations in tests observable, never certified.
        reference_version = "unknown_unverified"
    return ReferenceBandwidth(h=float(vals[0]), b=float(vals[2]),
                              selector=REFERENCE_SELECTORS[selector],
                              reference_version=reference_version)
