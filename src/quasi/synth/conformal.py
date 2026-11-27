"""Chernozhukov, Wuthrich & Zhu (2021, JASA) Moving-Block Conformal Inference for SCM/SDiD.

Provides finite-sample valid prediction/confidence intervals for single-treated-unit
synthetic control and SDiD estimators under weak time-series stationarity of the
synthetic control residuals (requiring no large number of treated units).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from quasi.errors import InsufficientSupportError, InvalidDesignError


@dataclass(frozen=True)
class ConformalIntervalResult:
    """Conformal confidence interval for the average post-treatment gap."""

    estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    p_value_null_zero: float
    n_permutations: int
    method: str = "moving_block_conformal_cwz2021"


def conformal_sc_interval(
    actual: np.ndarray,
    donors: np.ndarray,
    split: int,
    fit_synthetic_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    alpha: float = 0.05,
    grid_points: int = 81,
    point_estimate: float | None = None,
) -> ConformalIntervalResult:
    """Compute CWZ (2021) moving-block conformal p-value (H0: tau=0) and inverted CI.

    Parameters
    ----------
    actual : (T,) treated unit outcome trajectory.
    donors : (N0, T) donor outcome matrix.
    split : index of first post-treatment period (T0).
    fit_synthetic_fn : callable(target_full, donors_full) -> synthetic_full (T,).
        Fits the counterfactual trajectory using the null-adjusted full series (T periods).
    alpha : significance level in (0, 1).
    grid_points : number of grid evaluations around the point estimate to invert p(tau) >= alpha.
    point_estimate : optional estimator ATT to center the inversion grid and report as ``estimate``.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        raise InvalidDesignError("alpha must lie strictly between 0 and 1")
    y1 = np.asarray(actual, dtype=float).reshape(-1)
    y0 = np.asarray(donors, dtype=float)
    t_total = len(y1)
    t_post = t_total - split
    if t_post < 1 or split < 3:
        raise InsufficientSupportError("Conformal inference requires >=3 pre-periods and >=1 post-period")
    if 1.0 / t_total > alpha:
        raise InsufficientSupportError(
            f"Total periods T={t_total} cannot achieve level alpha={alpha:g} (min p = {1.0/t_total:.3f})"
        )

    def _p_value(tau0: float) -> float:
        y_null = y1.copy()
        y_null[split:] -= tau0
        synth_null = fit_synthetic_fn(y_null, y0)
        u = y_null - synth_null  # length T
        # Moving-block cyclic shifts of length t_post (CWZ 2021 Section 3)
        obs_stat = float(np.mean(np.abs(u[split:])))
        # Cyclic shifts k = 0, 1, ..., T-1
        ge_count = 0
        for shift in range(t_total):
            rolled = np.roll(u, shift)
            stat = float(np.mean(np.abs(rolled[split:])))
            if stat >= obs_stat - 1e-12:
                ge_count += 1
        return float(ge_count / t_total)

    synth_base = fit_synthetic_fn(y1, y0)
    est = float(np.mean((y1 - synth_base)[split:])) if point_estimate is None else float(point_estimate)
    pre_scale = max(float(np.std((y1 - synth_base)[:split])), float(np.std(y1[:split])) * 0.1, 1e-4)
    p_zero = _p_value(0.0)

    # Expand search bracket until endpoints reject H0 (or max expansion reached)
    half_width = max(4.0 * pre_scale, abs(est) * 2.0, 1.0)
    for _ in range(5):
        if _p_value(est - half_width) < alpha and _p_value(est + half_width) < alpha:
            break
        half_width *= 2.0

    grid = np.linspace(est - half_width, est + half_width, grid_points)
    pvals = np.array([_p_value(float(g)) for g in grid])
    accepted = grid[pvals >= alpha]
    if len(accepted) == 0:
        # Empty acceptance set (e.g., strongly time-varying post-treatment effects reject constant H0 everywhere)
        ci_lo = float("nan")
        ci_hi = float("nan")
    else:
        ci_lo = float(accepted[0])
        ci_hi = float(accepted[-1])

    return ConformalIntervalResult(
        estimate=est,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        alpha=float(alpha),
        p_value_null_zero=p_zero,
        n_permutations=t_total,
    )
