"""Frozen comparison fixture contract for *the named* phase-7 RDD benchmark.

Authenticating the bytes prevents accidental silent fixture replacement. A
checksum alone cannot authenticate independently produced *reference* output;
reference parity remains separately gated on actually running rdrobust/R.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

PHASE7_RDD_SHA256 = "cd6f007fb4d3e0fa01a97e8d8bee8c994516d725d4227ec1d8807099b2814854"


def load_phase7_fixture(path: Path) -> tuple[pd.DataFrame, str]:
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != PHASE7_RDD_SHA256:
        raise ValueError("RDD parity fixture integrity failed: SHA-256 does not match the frozen benchmark")
    data = pd.read_csv(path)
    if len(data) != 1800 or list(data.columns) != ["x", "y", "d"]:
        raise ValueError("RDD parity fixture schema mismatch")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("RDD parity fixture contains nonfinite observations")
    if not data["d"].isin((0, 1)).all():
        raise ValueError("RDD parity fixture treatment is not binary")
    if not (data.x.lt(0).any() and data.x.ge(0).any()):
        raise ValueError("RDD parity fixture has no cutoff support")
    return data, digest
