"""
RDD bandwidth selection:
- Native scale-invariant CCT-style MSE-optimal (``mserd``) and CER-optimal
  (``cerrd``) local-polynomial bandwidth selectors (Calonico, Cattaneo &
  Titiunik 2014; Calonico, Cattaneo & Farrell 2020).
- Legacy ``select_cct_bandwidth`` retained for compatibility with scale-invariant
  regularization and order-correct N^{-1/(2p+3)} rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from quasi.rdd.kernels import KERNEL_C_K, KERNEL_REGISTRY


@dataclass(frozen=True)
class NativeBandwidthSelection:
    h: float
    b: float
    selector: str
    density_at_cutoff: float
    bias_estimate: float
    variance_constant: float
    regularization_term: float


def _boundary_kernel_constants(kernel: str, order: int) -> tuple[float, float]:
    """Exact asymptotic boundary variance V_{p,K} and bias B_{p,K} on [0, 1] via Gauss-Legendre."""
    k_fn = KERNEL_REGISTRY[kernel.lower()]
    # 64-point Gauss-Legendre quadrature on [0, 1] is exact for polynomials up to degree 127
    nodes, weights = np.polynomial.legendre.leggauss(64)
    u = 0.5 * (nodes + 1.0)
    w_q = 0.5 * weights
    ku = k_fn(u)
    p1 = order + 1
    powers = np.vstack([u**j for j in range(p1)])  # (p+1) x Q
    gamma = (powers * (ku * w_q)) @ powers.T
    psi = (powers * (ku * ku * w_q)) @ powers.T
    u_p1 = u**p1
    lambda_vec = (powers * (ku * w_q)) @ u_p1
    inv_gamma = np.linalg.solve(gamma, np.eye(p1))
    e0 = np.zeros(p1)
    e0[0] = 1.0
    v_const = float(e0 @ inv_gamma @ psi @ inv_gamma @ e0)
    b_const = float(e0 @ inv_gamma @ lambda_vec)
    return v_const, b_const


def _side_pilot_fit(
    x_side: np.ndarray,
    y_side: np.ndarray,
    pilot_h: float,
    degree: int,
    kernel: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Local polynomial WLS fit on one side returning (beta, HC0_cov, residual_var)."""
    u = x_side / pilot_h
    w = KERNEL_REGISTRY[kernel.lower()](u)
    active = w > 0.0
    if int(active.sum()) < degree + 3 or np.unique(x_side[active]).size < degree + 2:
        raise ValueError("Insufficient distinct observations in pilot bandwidth")
    # Fit on dimensionless scaled coordinate u = x / pilot_h for exact scale invariance, then rescale back.
    z = np.vander(u[active], N=degree + 1, increasing=True)
    sw = np.sqrt(w[active])
    zw = z * sw[:, None]
    yw = y_side[active] * sw
    beta_u, _, rank, _ = np.linalg.lstsq(zw, yw, rcond=None)
    if rank < degree + 1:
        raise ValueError("Pilot local polynomial is rank-deficient")
    resid = y_side[active] - z @ beta_u
    _, r_mat = np.linalg.qr(zw, mode="reduced")
    inv_r = np.linalg.solve(r_mat, np.eye(degree + 1))
    bread = inv_r @ inv_r.T
    scores = z * (w[active] * resid)[:, None]
    cov_u = bread @ (scores.T @ scores) @ bread
    scale_vec = pilot_h ** (-np.arange(degree + 1, dtype=float))
    beta = beta_u * scale_vec
    cov = (scale_vec[:, None] * cov_u) * scale_vec[None, :]
    sigma2 = float(np.sum(w[active] * (resid**2)) / max(float(np.sum(w[active])), 1e-12))
    return beta, cov, max(sigma2, 1e-12)


def select_mse_bandwidth(
    X: np.ndarray,
    Y: np.ndarray,
    cutoff: float,
    *,
    kernel: str = "triangular",
    order: int = 1,
    selector: str = "mserd",
) -> NativeBandwidthSelection:
    """Native scale-invariant CCT-style MSE-optimal (``mserd``) or CER-optimal (``cerrd``) selector.

    Features:
    - Exact boundary kernel constants via Gauss-Legendre quadrature.
    - Scale-invariant CCT regularization term R_B derived from the variance of the
      pilot (p+1)-order derivative estimator (invariant to arbitrary rescaling of X or Y).
    - Order-specific asymptotic rate N^{-1/(2p+3)} for MSE and CCF (2020) CER scaling.
    """
    x = np.asarray(X, dtype=float).reshape(-1)
    y = np.asarray(Y, dtype=float).reshape(-1)
    n = len(x)
    x_c = x - float(cutoff)
    sd_x = float(np.std(x_c))
    if sd_x <= 1e-12:
        raise ValueError("Running variable has zero variance.")

    # Scale-invariant Silverman-style pilot bandwidth for density and order-(p+1) curvature
    iqr_x = float(np.subtract(*np.percentile(x_c, [75, 25])))
    spread = min(sd_x, iqr_x / 1.349) if iqr_x > 0 else sd_x
    q = order + 1
    # Rate for estimating (p+1)th derivative is N^{-1/(2q+3)}
    pilot_b = 2.5 * spread * (n ** (-1.0 / (2 * q + 3)))
    left_all = x_c[x_c < 0.0]
    right_all = x_c[x_c >= 0.0]
    if len(left_all) < q + 3 or len(right_all) < q + 3:
        raise ValueError("Insufficient observations on one or both sides of the cutoff")

    # Ensure pilot window covers at least max(20, q+5) observations on each side
    min_side = min(max(20, q + 5), len(left_all) - 1, len(right_all) - 1)
    floor_b = max(
        float(np.partition(np.abs(left_all), min_side)[min_side]),
        float(np.partition(np.abs(right_all), min_side)[min_side]),
    )
    pilot_b = max(pilot_b, floor_b * 1.05)

    # Local polynomial fits of order q = p + 1 on each side for curvature & conditional variance
    beta_r, cov_r, sig2_r = _side_pilot_fit(x_c[x_c >= 0.0], y[x_c >= 0.0], pilot_b, q, kernel)
    beta_l, cov_l, sig2_l = _side_pilot_fit(x_c[x_c < 0.0], y[x_c < 0.0], pilot_b, q, kernel)

    # Boundary density f(c) using triangular kernel inside pilot window (scale-invariant)
    h_dens = max(1.84 * spread * (n ** (-0.2)), floor_b)
    w_dens = KERNEL_REGISTRY["triangular"](x_c / h_dens)
    f_c = float(np.sum(w_dens) / (n * h_dens))
    f_c = max(f_c, 1e-8 / spread)

    v_const, b_const = _boundary_kernel_constants(kernel, order)
    # Bias term for (tau_hat): B = b_const * (beta_r[q] - (-1)^(q) * beta_l[q])
    # Note: for x_c < 0, u = x_c/h < 0, so E[y|x] = sum beta_j x^j => left boundary integral has (-1)^(p+1)
    bias_jump = b_const * (float(beta_r[q]) - ((-1.0) ** q) * float(beta_l[q]))
    # CCT scale-invariant regularization: 3 * Var(bias_jump_hat)
    var_bias_jump = (b_const**2) * (float(cov_r[q, q]) + float(cov_l[q, q]))
    reg_term = 3.0 * max(var_bias_jump, 0.0)

    v_total = v_const * (sig2_r + sig2_l) / f_c
    denom = 2.0 * q * (bias_jump**2 + reg_term)
    if not np.isfinite(denom) or denom <= 1e-18:
        h_mse = pilot_b
    else:
        h_mse = float(((v_total / denom) ** (1.0 / (2 * order + 3))) * (n ** (-1.0 / (2 * order + 3))))

    if selector == "cerrd":
        # Calonico, Cattaneo & Farrell (2020) CER-optimal bandwidth adjustment
        h_opt = h_mse * (n ** (-order / ((2 * order + 3) * (order + 3))))
    elif selector == "mserd":
        h_opt = h_mse
    else:
        raise ValueError(f"Unsupported native RDD selector: {selector!r}")

    min_support = min(max(12, order + 4), len(left_all) - 1, len(right_all) - 1)
    min_h = max(
        float(np.partition(np.abs(left_all), min_support)[min_support]),
        float(np.partition(np.abs(right_all), min_support)[min_support]),
    )
    max_h = float(min(np.max(np.abs(left_all)), np.max(np.abs(right_all))))
    if min_h * 1.02 > max_h:
        max_h = float(max(np.max(np.abs(left_all)), np.max(np.abs(right_all))))
    h_opt = float(np.clip(h_opt, min_h * 1.02, max_h))
    b_opt = float(np.clip(max(pilot_b, h_opt), h_opt, max_h))
    return NativeBandwidthSelection(
        h=h_opt,
        b=b_opt,
        selector=selector,
        density_at_cutoff=f_c,
        bias_estimate=bias_jump,
        variance_constant=v_total,
        regularization_term=reg_term,
    )


def select_cct_bandwidth(
    X: np.ndarray,
    Y: np.ndarray,
    cutoff: float,
    kernel: str = "triangular",
    order: int = 1,
) -> Tuple[float, float, float]:
    """Scale-invariant heuristic bandwidth selector (legacy API compatibility).

    Fixed from earlier pre-release versions to be strictly invariant to linear
    rescaling of X and Y and to apply the order-appropriate N^{-1/(2p+3)} rate.
    Returns ``(h_optimal, pilot_bandwidth_b, marginal_density_f_c)``.
    """
    N = len(X)
    x_centered = np.asarray(X, dtype=float) - cutoff
    y_arr = np.asarray(Y, dtype=float)
    sd_x = float(np.std(x_centered))
    if sd_x < 1e-12:
        raise ValueError("Running variable has zero variance.")

    # Normalize x internally to unit SD so every heuristic threshold is scale-invariant
    z_x = x_centered / sd_x
    pilot_b_z = 2.58 * (N ** (-1.0 / 7.0))

    pilot_mask = np.abs(z_x) <= pilot_b_z
    if np.sum(pilot_mask & (z_x >= 0)) < 15 or np.sum(pilot_mask & (z_x < 0)) < 15:
        pilot_b_z = 3.0 * (N ** (-1.0 / 7.0))
        pilot_mask = np.abs(z_x) <= pilot_b_z

    right_mask = (z_x >= 0) & pilot_mask
    Z_r = np.column_stack([np.ones(np.sum(right_mask)), z_x[right_mask], z_x[right_mask] ** 2])
    if np.linalg.matrix_rank(Z_r) < 3 or len(Z_r) < 4:
        raise ValueError("Heuristic bandwidth pilot lacks three distinct right-side running scores")
    beta_r = np.linalg.lstsq(Z_r, y_arr[right_mask], rcond=None)[0]
    m2_plus = 2.0 * beta_r[2]
    resid_r = y_arr[right_mask] - Z_r @ beta_r
    sigma2_plus = float(np.var(resid_r)) if len(resid_r) > 3 else float(np.var(y_arr[right_mask]))

    left_mask = (z_x < 0) & pilot_mask
    Z_l = np.column_stack([np.ones(np.sum(left_mask)), z_x[left_mask], z_x[left_mask] ** 2])
    if np.linalg.matrix_rank(Z_l) < 3 or len(Z_l) < 4:
        raise ValueError("Heuristic bandwidth pilot lacks three distinct left-side running scores")
    beta_l = np.linalg.lstsq(Z_l, y_arr[left_mask], rcond=None)[0]
    m2_minus = 2.0 * beta_l[2]
    resid_l = y_arr[left_mask] - Z_l @ beta_l
    sigma2_minus = float(np.var(resid_l)) if len(resid_l) > 3 else float(np.var(y_arr[left_mask]))

    n_in_pilot = np.sum(pilot_mask)
    f_c_z = max(float(n_in_pilot / (2.0 * pilot_b_z * N)), 1e-6)

    # Strictly scale-invariant curvature regularization in standardized z_x units
    sd_y = max(float(np.std(y_arr)), 1e-12)
    curvature_diff = abs(m2_plus - m2_minus)
    curvature_reg = max(curvature_diff, 0.08 * sd_y)

    ck = KERNEL_C_K.get(kernel.lower(), 1.43097)
    rate = 1.0 / (2 * order + 3)
    numerator = max(sigma2_plus + sigma2_minus, 1e-12 * sd_y * sd_y)
    denominator = f_c_z * (curvature_reg**2)

    h_z = float(ck * ((numerator / denominator) ** rate) * (N ** (-rate)))
    h_opt = h_z * sd_x
    pilot_b = pilot_b_z * sd_x
    f_c = f_c_z / sd_x

    min_h = float(np.partition(np.abs(x_centered), min(25, N - 1))[min(25, N - 1)])
    max_h = float(np.max(np.abs(x_centered)))
    h_opt = float(np.clip(h_opt, min_h, max_h))
    return h_opt, pilot_b, f_c
