"""Regression Discontinuity Design module for quasi."""

from __future__ import annotations

from quasi.rdd.bandwidth import select_cct_bandwidth
from quasi.rdd.kernels import epanechnikov_kernel, triangular_kernel, uniform_kernel
from quasi.rdd.regression import (
    Discontinuity,
    FuzzyRDD,
    RDDResult,
    RDDRobustnessTable,
    SharpRDD,
)

__all__ = [
    "Discontinuity",
    "SharpRDD",
    "FuzzyRDD",
    "RDDResult",
    "RDDRobustnessTable",
    "select_cct_bandwidth",
    "triangular_kernel",
    "epanechnikov_kernel",
    "uniform_kernel",
]