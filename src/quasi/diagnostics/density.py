"""
Exploratory binned running-variable density discontinuity check.
This approximation is NOT a validated McCrary (2008) estimator.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy import stats

from quasi.base import DiagnosticStatus, DiagnosticTest


def _triangular_kernel(u: np.ndarray) -> np.ndarray:
    abs_u = np.abs(u)
    return np.where(abs_u <= 1.0, 1.0 - abs_u, 0.0)


def mccrary_density_test(
    X: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    bin_width: Optional[float] = None,
) -> DiagnosticTest:
    """
    Computes an unvalidated density-jump screen for exploratory diagnostics only.

    H0: Continuity of running variable density f(x) at cutoff c.
    H1: Jump/discontinuity in density (indicates sorting or manipulation).
    """
    x_clean = X[~np.isnan(X)]
    N = len(x_clean)
    sd_x = float(np.std(x_clean))
    iqr_x = float(np.subtract(*np.percentile(x_clean, [75, 25])))
    spread = min(sd_x, iqr_x / 1.349) if iqr_x > 0 else sd_x

    # 1. Optimal fine bin width b_bins = 2 * spread * N^(-1/2)
    b = bin_width if bin_width is not None else float(2.0 * spread * (N ** (-0.5)))
    b = max(b, 1e-4)

    # 2. Construct non-overlapping bins strictly partitioned on either side of cutoff
    x_min = float(np.min(x_clean))
    x_max = float(np.max(x_clean))

    left_edges = np.arange(cutoff, x_min - b, -b)[::-1]
    right_edges = np.arange(cutoff, x_max + b, b)
    all_edges = np.unique(np.concatenate([left_edges, right_edges]))

    counts, edges = np.histogram(x_clean, bins=all_edges)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    densities = counts / (N * b)

    # 3. Kernel bandwidth for density local polynomial
    h = bandwidth if bandwidth is not None else float(1.5 * spread * (N ** (-0.2)))
    h = max(h, 2.0 * b)

    # 4. Local linear regression on bin midpoints separately left and right
    def estimate_boundary_density(
        x_mids: np.ndarray, y_dens: np.ndarray, side: str
    ) -> Tuple[float, float]:
        if side == "right":
            mask = (x_mids >= cutoff) & (x_mids <= cutoff + h)
        else:
            mask = (x_mids < cutoff) & (x_mids >= cutoff - h)

        if np.sum(mask) < 4:
            return 1e-4, 1.0

        x_eval = x_mids[mask] - cutoff
        y_eval = y_dens[mask]
        weights = _triangular_kernel(x_eval / h)
        active = weights > 0.0
        if int(np.sum(active)) < 3:
            return 1e-4, 1.0
        x_eval = x_eval[active]
        y_eval = y_eval[active]
        weights = weights[active]

        Z = np.column_stack([np.ones_like(x_eval), x_eval])
        sqrt_w = np.sqrt(weights)
        Zw = Z * sqrt_w[:, None]
        yw = y_eval * sqrt_w

        beta, _, _, _ = np.linalg.lstsq(Zw, yw, rcond=None)
        f_hat = float(max(beta[0], 1e-6))

        # O(M) HC0 sandwich asymptotic variance via QR without dense diagonal matrices
        residuals = y_eval - Z @ beta
        _, r_mat = np.linalg.qr(Zw, mode="reduced")
        try:
            inv_r = np.linalg.solve(r_mat, np.eye(r_mat.shape[0]))
            bread = inv_r @ inv_r.T
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(Zw.T @ Zw)
        scores = Z * (weights * residuals)[:, None]
        Omega = scores.T @ scores
        var_f = float((bread @ Omega @ bread)[0, 0])

        return f_hat, var_f

    f_right, var_right = estimate_boundary_density(midpoints, densities, side="right")
    f_left, var_left = estimate_boundary_density(midpoints, densities, side="left")

    # 5. Log Discontinuity: theta = ln(f_+) - ln(f_-)
    theta = float(np.log(f_right) - np.log(f_left))
    # Delta method standard error: SE(theta) = sqrt( var(f_+) / f_+^2 + var(f_-) / f_-^2 )
    se_theta = float(np.sqrt((var_right / (f_right**2)) + (var_left / (f_left**2))))

    z_stat = theta / se_theta if se_theta > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    # Evaluation status
    # This test is NOT independently validated or calibrated. It may identify
    # suspicious bin patterns, but must never report a certifying FAIL/PASS.
    status = DiagnosticStatus.WARN

    message = (
        "No large jump flagged by the exploratory binned screen; density continuity and absence of manipulation remain unverified."
        if p_val >= 0.10
        else (
            f"Possible density sorting pattern (theta = {theta:.3f}, exploratory p = {p_val:.3f})."
            if p_val >= 0.05
            else f"Possible density sorting pattern (theta = {theta:.3f}, exploratory p = {p_val:.3f}); significance unvalidated."
        )
    )

    remedy = (
        None
        if p_val >= 0.10
        else "Test for self-selection or institutional score clustering. Consider a Donut-Hole RDD."
    )

    return DiagnosticTest(
        name="Exploratory Density-Jump Screen (not McCrary)",
        status=status,
        statistic_name="Log Density Jump",
        statistic_value=theta,
        p_value=p_val,
        threshold="exploratory binned screen; p-values uncalibrated",
        message=message,
        remedy=remedy,
    )


def cjm_density_test(
    X: np.ndarray,
    cutoff: float,
    *,
    p: int = 2,
    bandwidth: Optional[float] = None,
) -> DiagnosticTest:
    """Cattaneo, Jansson & Ma (2020, JASA) local-polynomial CDF density continuity test.

    Estimates f_+(c) and f_-(c) as the first derivative of the empirical CDF via
    local polynomial regression of order p (default p=2, triangular kernel) with
    no arbitrary histogram binning. Tests H0: f_+(c) = f_-(c).
    """
    x_clean = np.asarray(X, dtype=float)
    x_clean = x_clean[np.isfinite(x_clean)]
    n = len(x_clean)
    if n < 30:
        return DiagnosticTest(
            name="CJM (2020) Density Continuity Test",
            status=DiagnosticStatus.WARN,
            statistic_name="T_CJM",
            statistic_value=0.0,
            p_value=None,
            threshold="requires >= 30 observations",
            message="Too few observations for local-polynomial density continuity testing.",
        )

    x_c = np.sort(x_clean - float(cutoff))
    sd_x = float(np.std(x_c))
    iqr_x = float(np.subtract(*np.percentile(x_c, [75, 25])))
    spread = min(sd_x, iqr_x / 1.349) if iqr_x > 0 else sd_x
    if spread <= 1e-12:
        return DiagnosticTest(
            name="CJM (2020) Density Continuity Test",
            status=DiagnosticStatus.WARN,
            statistic_name="T_CJM",
            statistic_value=0.0,
            p_value=None,
            threshold="requires nondegenerate running variable",
            message="Running variable has zero spread; density continuity test unavailable.",
        )

    # Order-p local-polynomial CDF derivative MSE rate: h ~ n^{-1/(2p+1)}
    h = float(bandwidth) if bandwidth is not None else float(2.2 * spread * (n ** (-1.0 / (2 * p + 1))))
    # Empirical CDF evaluated at sorted order statistics (midpoint rank for symmetry)
    ecdf = (np.arange(1, n + 1, dtype=float) - 0.5) / n

    def _fit_side(side: str) -> Tuple[float, float, int]:
        mask = (x_c >= 0.0) & (x_c <= h) if side == "right" else (x_c < 0.0) & (x_c >= -h)
        xs = x_c[mask]
        fs = ecdf[mask]
        if len(xs) < p + 3 or np.unique(xs).size < p + 2:
            return float("nan"), float("nan"), len(xs)
        u = xs / h
        w = _triangular_kernel(u)
        active = w > 0.0
        u = u[active]
        fs = fs[active]
        w = w[active]
        if len(u) < p + 3:
            return float("nan"), float("nan"), len(u)
        z = np.vander(u, N=p + 1, increasing=True)
        sw = np.sqrt(w)
        zw = z * sw[:, None]
        beta, _, rank, _ = np.linalg.lstsq(zw, fs * sw, rcond=None)
        if rank < p + 1:
            return float("nan"), float("nan"), len(u)
        # First derivative of F at 0 in original x units: f(c) = beta[1] / h
        f_hat = float(beta[1] / h)
        # Asymptotic variance of CJM local-polynomial density estimator:
        # Var(f_hat) = (f_hat / (n * h)) * e_1' Gamma^{-1} S Gamma^{-1} e_1
        # Compute exact boundary Gram/covariance integrals on [0, 1] via quadrature
        nodes, q_w = np.polynomial.legendre.leggauss(48)
        g_u = 0.5 * (nodes + 1.0)
        g_w = 0.5 * q_w
        k_g = _triangular_kernel(g_u)
        p_mat = np.vstack([g_u**j for j in range(p + 1)])
        gamma = (p_mat * (k_g * g_w)) @ p_mat.T
        # Exact Brownian-bridge covariance kernel min(u, v) for empirical CDF increments:
        # S_{ij} = int_0^1 int_0^1 u^i v^j K(u) K(v) min(u, v) du dv
        min_uv = np.minimum(g_u[:, None], g_u[None, :])
        kw = k_g * g_w
        s_mat = (p_mat * kw) @ min_uv @ (p_mat * kw).T
        inv_g = np.linalg.solve(gamma, np.eye(p + 1))
        e1 = np.zeros(p + 1)
        e1[1] = 1.0
        # For empirical CDF derivative: Var(f_hat_side) ~ (f_side / (n * h)) * (e1' G^-1 S G^-1 e1)
        v_const = float(e1 @ inv_g @ s_mat @ inv_g @ e1)
        var_f = max(f_hat, 1e-8) * v_const / (n * h)
        return max(f_hat, 0.0), max(var_f, 1e-16), len(u)

    f_r, var_r, n_r = _fit_side("right")
    f_l, var_l, n_l = _fit_side("left")
    if not (np.isfinite(f_r) and np.isfinite(f_l) and np.isfinite(var_r) and np.isfinite(var_l)):
        return DiagnosticTest(
            name="CJM (2020) Density Continuity Test",
            status=DiagnosticStatus.WARN,
            statistic_name="T_CJM",
            statistic_value=0.0,
            p_value=None,
            threshold="insufficient local support",
            message=f"Insufficient local support around cutoff (n_left={n_l}, n_right={n_r}).",
        )

    se_diff = float(np.sqrt(var_r + var_l))
    t_stat = float((f_r - f_l) / se_diff) if se_diff > 0 else 0.0
    p_val = float(2.0 * stats.norm.sf(abs(t_stat)))
    status = DiagnosticStatus.PASS if p_val >= 0.10 else (DiagnosticStatus.WARN if p_val >= 0.05 else DiagnosticStatus.FAIL)
    msg = (
        f"No statistically detectable density jump (f_right={f_r:.4f}, f_left={f_l:.4f}, T={t_stat:.3f}, p={p_val:.4f}); "
        "non-rejection does not prove absence of manipulation."
        if p_val >= 0.05
        else f"Significant density discontinuity at cutoff (f_right={f_r:.4f}, f_left={f_l:.4f}, T={t_stat:.3f}, p={p_val:.4f})."
    )
    return DiagnosticTest(
        name="CJM (2020) Density Continuity Test",
        status=status,
        statistic_name="T_CJM",
        statistic_value=t_stat,
        p_value=p_val,
        threshold="p >= 0.05 (CJM local-polynomial CDF test)",
        message=msg,
        remedy=None if p_val >= 0.05 else "Investigate sorting/heaping around cutoff and run donut_sensitivity().",
    )