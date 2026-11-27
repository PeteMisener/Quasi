"""Shared numerical and inference utilities (rank-checked, fail-closed)."""

from quasi.inference.covariance import (
    CR2Covariance,
    EWCCovariance,
    MultiwayClusterCovariance,
    bartlett_lag,
    cluster_covariance,
    cr2_covariance,
    ewc_har_covariance,
    hac_covariance,
    hc_covariance,
    multiway_cluster_covariance,
    prewhitened_hac_covariance,
)
from quasi.inference.linear import LinearFit, least_squares, two_stage_least_squares
from quasi.inference.optimization import (
    SimplexSolution,
    checked_simplex_qp,
    project_simplex,
    solve_simplex_regression,
)
from quasi.inference.resampling import (
    WildBootstrapResult,
    bootstrap_indices,
    percentile_interval,
    wild_cluster_bootstrap,
)
from quasi.inference.serial import PraisWinstenFit, prais_winsten_ar1, profile_ml_ar1, reml_ar1

__all__ = [
    "CR2Covariance",
    "EWCCovariance",
    "LinearFit",
    "MultiwayClusterCovariance",
    "PraisWinstenFit",
    "SimplexSolution",
    "WildBootstrapResult",
    "bartlett_lag",
    "bootstrap_indices",
    "checked_simplex_qp",
    "cluster_covariance",
    "cr2_covariance",
    "ewc_har_covariance",
    "hac_covariance",
    "hc_covariance",
    "least_squares",
    "multiway_cluster_covariance",
    "percentile_interval",
    "prais_winsten_ar1",
    "prewhitened_hac_covariance",
    "profile_ml_ar1",
    "project_simplex",
    "reml_ar1",
    "solve_simplex_regression",
    "two_stage_least_squares",
    "wild_cluster_bootstrap",
]
