"""Rambachan & Roth (2023, ReStud) HonestDiD Pre-Trends Sensitivity Analysis.

Computes robust confidence intervals for post-treatment linear contrasts under
bounded violations of parallel trends:
- Relative Magnitudes (Delta^RM(M_bar)): max post-period trend violation is at
  most M_bar times the maximum pre-period violation.
- Smoothness (Delta^SD(M)): second differences of the trend violation are
  bounded in magnitude by M.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

from quasi.errors import InsufficientSupportError, InvalidDesignError


@dataclass(frozen=True)
class HonestDiDResult:
    """Robust sensitivity grid and breakdown value for parallel-trends violations."""

    method: str
    alpha: float
    target_estimate: float
    target_se: float
    breakdown_value: float | None
    table: pd.DataFrame


def _smoothness_bounds(
    beta_pre: np.ndarray,
    l_post: np.ndarray,
    m_val: float,
) -> tuple[float, float]:
    """Sharp worst-case bias bounds [min l'delta_post, max l'delta_post] under Delta^SD(M)."""
    k_pre = len(beta_pre)
    k_post = len(l_post)
    # Full trend vector delta = [delta_{-k_pre}, ..., delta_{-2}, 0 (ref -1), delta_0, ..., delta_{k_post-1}]
    # Pre-period entries delta_pre are fixed at beta_pre (plug-in / identified set center);
    # we optimize over free vector d_post of length k_post subject to |Delta^2 delta_t| <= M.
    n_tot = k_pre + 1 + k_post
    # Build second-difference rows on full vector: delta_{t+1} - 2*delta_t + delta_{t-1}
    a_ub_rows = []
    b_ub_vals = []
    # Map full index j in [0, n_tot-1]:
    # j < k_pre: fixed at beta_pre[j]
    # j == k_pre: fixed at 0.0 (reference period)
    # j > k_pre: free variable d_post[j - k_pre - 1]
    for t in range(1, n_tot - 1):
        row = np.zeros(k_post, dtype=float)
        const = 0.0
        for idx, coeff in ((t - 1, 1.0), (t, -2.0), (t + 1, 1.0)):
            if idx < k_pre:
                const += coeff * float(beta_pre[idx])
            elif idx == k_pre:
                pass  # reference normalized to 0
            else:
                row[idx - k_pre - 1] += coeff
        if np.any(row != 0.0):
            # row @ d_post + const <= M  and  -row @ d_post - const <= M
            a_ub_rows.append(row)
            b_ub_vals.append(m_val - const)
            a_ub_rows.append(-row)
            b_ub_vals.append(m_val + const)

    a_ub = np.asarray(a_ub_rows, dtype=float)
    b_ub = np.asarray(b_ub_vals, dtype=float)
    res_min = linprog(l_post, A_ub=a_ub, b_ub=b_ub, bounds=(None, None), method="highs")
    res_max = linprog(-l_post, A_ub=a_ub, b_ub=b_ub, bounds=(None, None), method="highs")
    if not res_min.success or not res_max.success:
        raise InsufficientSupportError("HonestDiD smoothness LP failed to solve")
    return float(res_min.fun), float(-res_max.fun)


def _relative_magnitudes_bounds(
    beta_pre: np.ndarray,
    l_post: np.ndarray,
    m_bar: float,
) -> tuple[float, float]:
    """Sharp worst-case bias bounds under Delta^RM(M_bar): |delta_t - delta_{t-1}| <= M_bar * max_pre."""
    # Pre-period increments including step into normalized reference period 0
    pre_with_ref = np.append(beta_pre, 0.0)
    pre_diffs = np.abs(np.diff(pre_with_ref))
    max_pre = float(np.max(pre_diffs)) if len(pre_diffs) else 0.0
    step_bound = m_bar * max_pre

    k_post = len(l_post)
    # Post increments u_0 = delta_0 - 0, u_j = delta_j - delta_{j-1}, each |u_j| <= step_bound.
    # Then delta_r = sum_{j=0}^r u_j => l' delta_post = sum_{j=0}^{k_post-1} (sum_{r=j}^{k_post-1} l_r) u_j.
    cum_weights = np.cumsum(l_post[::-1])[::-1]
    max_bias = float(np.sum(np.abs(cum_weights)) * step_bound)
    return -max_bias, max_bias


def honest_did_sensitivity(
    event_times: Sequence[int],
    estimates: Sequence[float],
    covariance: np.ndarray,
    *,
    reference_event: int = -1,
    base_period: Literal["universal", "varying"] = "universal",
    method: Literal["relative_magnitudes", "smoothness"] = "relative_magnitudes",
    m_grid: Sequence[float] | None = None,
    post_weights: np.ndarray | None = None,
    alpha: float = 0.05,
) -> HonestDiDResult:
    """Compute HonestDiD (Rambachan & Roth 2023) robust CIs over a grid of M / M_bar.

    When ``base_period="varying"`` (adjacent first differences for pre-periods),
    pre-period estimates and their joint covariance are linearly cumulated back
    onto the universal reference-period scale before evaluating bounds. For
    ``method="smoothness"``, full pre/post covariance is propagated through the
    linear trend-extrapolation center shift.
    """
    if method not in ("relative_magnitudes", "smoothness"):
        raise InvalidDesignError("method must be 'relative_magnitudes' or 'smoothness'")
    if base_period not in ("universal", "varying"):
        raise InvalidDesignError("base_period must be 'universal' or 'varying'")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        raise InvalidDesignError("alpha must be in (0, 1)")

    et = np.asarray(event_times, dtype=int)
    est = np.asarray(estimates, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (len(et), len(et)) or len(est) != len(et):
        raise InvalidDesignError("event_times, estimates, and covariance shapes must agree")

    pre_idx = np.flatnonzero(et < reference_event)
    post_idx = np.flatnonzero(et > reference_event)
    if len(pre_idx) == 0:
        raise InsufficientSupportError("HonestDiD requires at least one pre-reference event period")
    if len(post_idx) == 0:
        raise InsufficientSupportError("HonestDiD requires at least one post-reference event period")

    # Sort chronologically within pre and post and verify contiguous integer spacing
    pre_idx = pre_idx[np.argsort(et[pre_idx])]
    post_idx = post_idx[np.argsort(et[post_idx])]
    pre_et = et[pre_idx]
    post_et = et[post_idx]
    expected_pre = np.arange(reference_event - len(pre_idx), reference_event, dtype=int)
    expected_post = np.arange(reference_event + 1, reference_event + 1 + len(post_idx), dtype=int)
    if not (np.array_equal(pre_et, expected_pre) and np.array_equal(post_et, expected_post)):
        raise InvalidDesignError(
            "HonestDiD requires contiguous integer event_times around reference_event without missing periods"
        )

    # Extract ordered (pre, post) vector and joint covariance
    ord_idx = np.concatenate([pre_idx, post_idx])
    k_pre = len(pre_idx)
    k_post = len(post_idx)
    beta_ord = est[ord_idx].copy()
    cov_ord = cov[np.ix_(ord_idx, ord_idx)].copy()

    if base_period == "varying":
        # Under varying base, pre-period j < ref reports E[Y_j - Y_{j-1}] (or ATT(g, t) vs t-1);
        # cumulative universal-relative levels satisfy delta_j = -sum_{s=j+1}^{ref} d_s.
        # Map adjacent pre-differences linearly via upper-triangular -1 matrix A_pre.
        a_mat = np.eye(k_pre + k_post, dtype=float)
        a_pre = np.zeros((k_pre, k_pre), dtype=float)
        for i in range(k_pre):
            for j in range(i, k_pre):
                a_pre[i, j] = -1.0
        a_mat[:k_pre, :k_pre] = a_pre
        beta_ord = a_mat @ beta_ord
        cov_ord = a_mat @ cov_ord @ a_mat.T

    if post_weights is None:
        l_post = np.zeros(k_post, dtype=float)
        # Default target: first non-anticipatory post-treatment period (e == 0 if present, else first post)
        zero_pos = np.flatnonzero(post_et == 0)
        l_post[int(zero_pos[0]) if len(zero_pos) else 0] = 1.0
    else:
        l_post = np.asarray(post_weights, dtype=float).reshape(-1)
        if len(l_post) != k_post or not np.isfinite(l_post).all():
            raise InvalidDesignError(f"post_weights must have length {k_post}")

    beta_pre = beta_ord[:k_pre]
    beta_post = beta_ord[k_pre:]
    cov_post = cov_ord[k_pre:, k_pre:]
    theta_hat = float(l_post @ beta_post)
    se_post = float(np.sqrt(max(float(l_post @ cov_post @ l_post), 0.0)))
    if se_post <= 1e-14:
        raise InsufficientSupportError("Target post-treatment contrast has zero variance")

    # For smoothness (Delta^SD), linear trend extrapolation from beta_pre is:
    # delta_post_lin[r] = -(r + 1) * beta_pre[-1] (since delta_ref = 0 and slope = 0 - beta_pre[-1]).
    # Propagate full pre+post covariance for (l_post' beta_post - l_post' delta_post_lin).
    w_smooth = np.zeros(k_pre + k_post, dtype=float)
    w_smooth[k_pre:] = l_post
    horizons = np.arange(1, k_post + 1, dtype=float)
    w_smooth[k_pre - 1] = float(l_post @ horizons)
    se_smooth = float(np.sqrt(max(float(w_smooth @ cov_ord @ w_smooth), 0.0)))

    grid = (
        [0.0, 0.5, 1.0, 1.5, 2.0]
        if m_grid is None and method == "relative_magnitudes"
        else ([0.0, 0.02, 0.05, 0.1, 0.2] if m_grid is None else list(m_grid))
    )
    if not grid or any(v < 0 or not np.isfinite(v) for v in grid):
        raise InvalidDesignError("m_grid must contain finite nonnegative values")

    # Exact Armstrong-Kolesar (2018) / Rambachan-Roth (2023) bias-aware critical value:
    # Solve P(|Z + b| <= cv) = 1 - alpha for Z ~ N(0, 1), which is sqrt(chi2_nc(1, b^2).ppf(1 - alpha)).
    def bias_aware_cv(bias_ratio: float) -> float:
        if bias_ratio <= 1e-12:
            return float(stats.norm.ppf(1.0 - alpha / 2.0))
        return float(np.sqrt(stats.ncx2.ppf(1.0 - alpha, df=1.0, nc=bias_ratio * bias_ratio)))

    rows = []
    breakdown: float | None = None
    active_se = se_smooth if method == "smoothness" and se_smooth > 1e-14 else se_post
    for m_val in sorted(float(v) for v in grid):
        if method == "relative_magnitudes":
            lo_b, hi_b = _relative_magnitudes_bounds(beta_pre, l_post, m_val)
        else:
            lo_b, hi_b = _smoothness_bounds(beta_pre, l_post, m_val)
        # Worst-case bias center & half-width around identified set
        center_shift = 0.5 * (lo_b + hi_b)
        max_half_bias = 0.5 * (hi_b - lo_b)
        cv = bias_aware_cv(max_half_bias / active_se)
        # Subtract bias from theta_hat (since E[theta_hat] = tau + l'delta_post)
        ci_low = (theta_hat - center_shift) - cv * active_se
        ci_high = (theta_hat - center_shift) + cv * active_se
        includes_zero = bool(ci_low <= 0.0 <= ci_high)
        if includes_zero and breakdown is None:
            breakdown = m_val
        rows.append({
            "M": m_val,
            "identified_set_low": theta_hat - hi_b,
            "identified_set_high": theta_hat - lo_b,
            "robust_ci_lower": ci_low,
            "robust_ci_upper": ci_high,
            "includes_zero": includes_zero,
        })

    return HonestDiDResult(
        method=method,
        alpha=float(alpha),
        target_estimate=theta_hat,
        target_se=active_se,
        breakdown_value=breakdown,
        table=pd.DataFrame(rows),
    )
