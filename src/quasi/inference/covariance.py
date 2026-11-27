"""Explicit sandwich & HAR covariance estimators:

- Heteroskedasticity-consistent: HC0, HC1, HC2, HC3
- Time-series HAC / HAR: Bartlett NW-HAC, Andrews-Monahan prewhitened HAC
  (Bartlett & Quadratic Spectral kernels), and Lazarus-Lewis-Stock-Watson (2018)
  Equal-Weighted Cosine (EWC) fixed-b HAR with exact t(B) degrees of freedom
- Clustered: One-way CR0/CR1, Bell-McCaffrey / Pustejovsky-Tipton CR2 with
  O(Np + G^2 p) Satterthwaite degrees of freedom, and Cameron-Gelbach-Miller
  multi-way clustering with explicit PSD audit metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from quasi.errors import InsufficientSupportError, InvalidDesignError, NotIdentifiedError
from quasi.inference.linear import _design, _outcome


def _inputs(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _design(x, "covariance design")
    u = _outcome(u, x.shape[0])
    # Avoid pseudoinverse; rank and conditioning failures are explicit.
    _, singular, _ = np.linalg.svd(x, full_matrices=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > 1e12:
        raise NotIdentifiedError("Covariance design is singular or ill-conditioned")
    # Use stable triangular QR solve for (X'X)^-1, not inv(X'X).
    _, r = np.linalg.qr(x, mode="reduced")
    inv_r = np.linalg.solve(r, np.eye(r.shape[0]))
    bread = inv_r @ inv_r.T
    return x, u, bread


def _sandwich(bread: np.ndarray, meat: np.ndarray) -> np.ndarray:
    cov = bread @ meat @ bread
    return (cov + cov.T) / 2.0


def _validate_groups(groups: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(groups)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise InvalidDesignError("groups must be a vector with one group label per row")
    try:
        import pandas as pd

        if pd.isna(arr).any():
            raise InvalidDesignError("cluster labels cannot be missing")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InvalidDesignError):
            raise
        raise InvalidDesignError("cluster labels must be nonmissing comparable values") from exc
    _, inverse = np.unique(arr, return_inverse=True)
    g = int(inverse.max() + 1)
    if g < 2:
        raise InsufficientSupportError("Cluster inference needs at least two independent groups")
    return inverse, g


def hc_covariance(x: np.ndarray, residuals: np.ndarray, kind: str = "HC0") -> np.ndarray:
    """OLS HC0/HC1/HC2/HC3; no implicit finite-sample factor for HC0."""
    x, u, bread = _inputs(x, residuals)
    n, p = x.shape
    kind = kind.upper()
    if kind not in {"HC0", "HC1", "HC2", "HC3"}:
        raise InvalidDesignError(f"Unsupported HC covariance variant: {kind}")
    scores = x * u[:, None]
    if kind in {"HC2", "HC3"}:
        leverage = np.einsum("ij,jk,ik->i", x, bread, x)
        if np.any(leverage >= 1 - 1e-12):
            raise InsufficientSupportError(f"{kind} requires all hat diagonal values < 1")
        scores = scores / ((1.0 - leverage) ** (0.5 if kind == "HC2" else 1.0))[:, None]
    meat = scores.T @ scores
    if kind == "HC1":
        meat *= n / (n - p)
    return _sandwich(bread, meat)


def bartlett_lag(n: int) -> int:
    """Historical NW94 default: floor(4 (n/100)^(2/9)), clipped to [1,n-1]."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise InvalidDesignError("Need at least two observations for lag selection")
    return max(1, min(n - 1, int(np.floor(4 * (n / 100.0) ** (2 / 9)))))


def hac_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    *,
    max_lag: int | None = None,
    small_sample: bool = False,
) -> np.ndarray:
    """Bartlett/Newey-West sandwich. max_lag=0 is the unadjusted HC0 meat.

    Input must be in temporal order with equally spaced observations (ensured
    by callers).
    """
    x, u, bread = _inputs(x, residuals)
    n, p = x.shape
    lag = bartlett_lag(n) if max_lag is None else max_lag
    if type(lag) is not int or lag < 0 or lag >= n:
        raise InvalidDesignError(f"HAC max_lag must be integer in [0, {n - 1}]")
    if type(small_sample) is not bool:
        raise InvalidDesignError("small_sample must be bool")
    scores = x * u[:, None]
    meat = scores.T @ scores
    for j in range(1, lag + 1):
        weights = (lag + 1 - j) / (lag + 1)
        cross = scores[j:].T @ scores[:-j]
        meat += weights * (cross + cross.T)
    if small_sample:
        meat *= n / (n - p)
    return _sandwich(bread, meat)


@dataclass(frozen=True)
class EWCCovariance:
    """Lazarus-Lewis-Stock-Watson (2018) Equal-Weighted Cosine HAR covariance."""

    covariance: np.ndarray
    n_cosines: int
    degrees_freedom: int


def ewc_har_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    *,
    n_cosines: int | None = None,
    finite_sample_projection: bool = True,
) -> EWCCovariance:
    """Equal-Weighted Cosine (EWC) fixed-b HAR covariance (LLSW 2018, JBES).

    Uses ``B = max(8, floor(0.4 * T^(2/3)))`` cosines by default (clipped to
    ``[2, T-1]``). When ``finite_sample_projection=True`` (default), applies an
    exact orthonormal-QR residual-projection correction ``A_ewc^{-1/2}`` so the
    variance estimator is unbiased under IID errors even for low-frequency
    deterministic trend/step regressors while preserving the exact LLSW (2018)
    fixed-b Student-t(B) asymptotic limit.
    """
    x, u, _ = _inputs(x, residuals)
    t_len, p = x.shape
    if t_len < 6:
        raise InsufficientSupportError("EWC HAR covariance requires at least 6 time periods")
    default_b = max(min(8, t_len - 1), min(t_len - 1, int(np.floor(0.4 * (t_len ** (2.0 / 3.0))))))
    b_val = default_b if n_cosines is None else n_cosines
    if type(b_val) is not int or b_val < 2 or b_val >= t_len:
        raise InvalidDesignError(f"n_cosines must be an integer in [2, {t_len - 1}]")

    # Work in orthonormal Q coordinates (X = Q R, Q'Q = I_p) for exact scale invariance.
    q_x, r_x = np.linalg.qr(x, mode="reduced")
    inv_r = np.linalg.solve(r_x, np.eye(p))

    scores_q = q_x * u[:, None]  # T x p
    grid = (np.arange(t_len, dtype=float) + 0.5) / t_len
    harmonics = np.arange(1, b_val + 1, dtype=float)
    basis = np.sqrt(2.0) * np.cos(np.pi * np.outer(harmonics, grid))  # B x T
    transformed = basis @ scores_q  # B x p
    meat_q = (transformed.T @ transformed) / b_val

    if finite_sample_projection:
        # Exact IID residual-projection expectation: A_ewc = (1/B) sum_j W_j' (I - Q Q') W_j
        a_ewc = np.zeros((p, p), dtype=float)
        for j in range(b_val):
            wj = q_x * basis[j, :, None]
            proj = q_x.T @ wj
            a_ewc += (wj.T @ wj) - (proj.T @ proj)
        a_ewc = 0.5 * (a_ewc + a_ewc.T) / b_val
        eigvals, eigvecs = np.linalg.eigh(a_ewc)
        if eigvals[0] > 1e-8:
            inv_half = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T
            meat_q = inv_half @ meat_q @ inv_half

    cov = inv_r @ meat_q @ inv_r.T
    cov = 0.5 * (cov + cov.T)
    return EWCCovariance(covariance=cov, n_cosines=b_val, degrees_freedom=b_val)


def _qs_weight(z: float) -> float:
    if z == 0.0:
        return 1.0
    x = 6.0 * np.pi * z / 5.0
    return float((25.0 / (12.0 * np.pi * np.pi * z * z)) * (np.sin(x) / x - np.cos(x)))


def prewhitened_hac_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    *,
    kernel: str = "bartlett",
    max_lag: int | None = None,
    prewhiten: bool = True,
    small_sample: bool = False,
) -> np.ndarray:
    """Andrews-Monahan (1992) VAR(1)-prewhitened HAC with Andrews (1991) plug-in bandwidth."""
    x, u, bread = _inputs(x, residuals)
    t_len, p = x.shape
    kern = kernel.lower()
    if kern not in ("bartlett", "qs"):
        raise InvalidDesignError("kernel must be 'bartlett' or 'qs'")
    scores = x * u[:, None]  # T x p
    # Standardize score columns so VAR(1) SVD damping & Andrews bandwidth are invariant to column scale
    col_scales = np.std(scores, axis=0)
    col_scales = np.where(col_scales > 1e-14, col_scales, 1.0)
    scores_std = scores / col_scales[None, :]

    # Step 1: Optional VAR(1) prewhitening of standardized score vectors with eigenvalue stabilization.
    a_std = np.zeros((p, p), dtype=float)
    if prewhiten and t_len > p + 4:
        s_lag = scores_std[:-1]
        s_cur = scores_std[1:]
        a_hat, _, _, _ = np.linalg.lstsq(s_lag, s_cur, rcond=None)
        a_hat = a_hat.T  # s_t = A s_{t-1} + e_t
        # SVD singular-value damping at 0.97 (Andrews & Monahan 1992) prevents unit-root explosion
        u_s, sing, vt_s = np.linalg.svd(a_hat, full_matrices=False)
        sing = np.minimum(sing, 0.97)
        a_std = (u_s * sing) @ vt_s
        work_std = s_cur - s_lag @ a_std.T
    else:
        work_std = scores_std

    m = work_std.shape[0]
    # Step 2: Andrews (1991) automatic bandwidth on (prewhitened) standardized scores if max_lag is None
    if max_lag is None:
        num = 0.0
        den = 0.0
        for col in range(p):
            v = work_std[:, col]
            v_denom = float(np.dot(v[:-1], v[:-1]))
            if v_denom <= 1e-14:
                continue
            rho_j = float(np.clip(np.dot(v[1:], v[:-1]) / v_denom, -0.95, 0.95))
            sig2_j = float(np.mean((v[1:] - rho_j * v[:-1]) ** 2))
            if kern == "bartlett":
                num += 4.0 * (rho_j**2) * (sig2_j**2) / (((1.0 - rho_j) ** 6) * ((1.0 + rho_j) ** 2))
                den += (sig2_j**2) / ((1.0 - rho_j) ** 4)
            else:
                num += 4.0 * (rho_j**2) * (sig2_j**2) / ((1.0 - rho_j) ** 8)
                den += (sig2_j**2) / ((1.0 - rho_j) ** 4)
        alpha_hat = num / den if den > 1e-14 else 0.0
        if kern == "bartlett":
            bw = 1.1447 * ((alpha_hat * m) ** (1.0 / 3.0)) if alpha_hat > 0 else float(bartlett_lag(m))
        else:
            bw = 1.3221 * ((alpha_hat * m) ** 0.2) if alpha_hat > 0 else float(bartlett_lag(m))
        bw = float(np.clip(bw, 1.0, m - 1.0))
    else:
        if type(max_lag) is not int or max_lag < 0 or max_lag >= m:
            raise InvalidDesignError(f"max_lag must be an integer in [0, {m - 1}]")
        bw = float(max_lag)

    meat_std = work_std.T @ work_std
    max_j = 0 if bw <= 0.0 else ((m - 1) if kern == "qs" else int(np.floor(bw)))
    for j in range(1, max_j + 1):
        weight = (1.0 - j / (bw + 1.0)) if kern == "bartlett" else _qs_weight(j / bw)
        if abs(weight) <= 1e-12:
            continue
        cross = work_std[j:].T @ work_std[:-j]
        meat_std += weight * (cross + cross.T)
    # Rescale to full T sum and recolor in standardized coordinates, then restore original column scales
    if prewhiten and m < t_len:
        meat_std *= t_len / m
        d_inv = np.linalg.solve(np.eye(p) - a_std, np.eye(p))
        meat_std = d_inv @ meat_std @ d_inv.T
    meat = (col_scales[:, None] * meat_std) * col_scales[None, :]
    if small_sample:
        meat *= t_len / (t_len - p)
    return _sandwich(bread, meat)


def cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    groups: np.ndarray,
    *,
    small_sample: bool = True,
) -> np.ndarray:
    """One-way CR0/CR1 sandwich using whole-cluster score sums."""
    x, u, bread = _inputs(x, residuals)
    n, p = x.shape
    inverse, g = _validate_groups(groups, n)
    if type(small_sample) is not bool:
        raise InvalidDesignError("small_sample must be bool")
    scores = np.zeros((g, p), dtype=float)
    np.add.at(scores, inverse, x * u[:, None])
    meat = scores.T @ scores
    if small_sample:
        meat *= (g / (g - 1)) * ((n - 1) / (n - p))
    return _sandwich(bread, meat)


@dataclass(frozen=True)
class CR2Covariance:
    """Bell-McCaffrey / Pustejovsky-Tipton CR2 covariance and Satterthwaite df."""

    covariance: np.ndarray
    n_clusters: int
    satterthwaite_df: np.ndarray  # per-parameter Satterthwaite degrees of freedom


def cr2_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    groups: np.ndarray,
    *,
    contrasts: np.ndarray | None = None,
) -> CR2Covariance:
    """Bell-McCaffrey (2002) / Pustejovsky-Tipton (2018) CR2 + Satterthwaite df.

    Computes exact cluster adjustment matrices A_g = (I - H_gg)^(-1/2) and exact
    O(N p + G^2 p) Satterthwaite effective degrees of freedom for every parameter
    (or user-supplied contrast rows) without constructing N x N matrices.
    """
    x, u, bread = _inputs(x, residuals)
    n, p = x.shape
    inverse, g = _validate_groups(groups, n)

    cluster_indices = [np.flatnonzero(inverse == j) for j in range(g)]
    # Factor bread = R^{-1} R^{-T} via QR(x) so H_gg = W_g W_g' with W_g = X_g R^{-1} (n_g x p).
    # Thin SVD W_g = U_g diag(s_g) V_g' gives exact A_g v = v + U_g (((1 - s_g^2)^(-1/2) - 1) * (U_g' v))
    # in O(n_g p^2) time and O(n_g p) memory without ever allocating n_g x n_g matrices.
    _, r_mat = np.linalg.qr(x, mode="reduced")
    r_inv = np.linalg.solve(r_mat, np.eye(p, dtype=float))
    u_factors: list[tuple[np.ndarray, np.ndarray]] = []
    x_blocks: list[np.ndarray] = []
    scores = np.zeros((g, p), dtype=float)

    def _apply_ag(u_g: np.ndarray, boost: np.ndarray, vec: np.ndarray) -> np.ndarray:
        return vec + u_g @ (boost * (u_g.T @ vec))

    for j, idx in enumerate(cluster_indices):
        xg = x[idx]  # n_g x p
        ug = u[idx]  # n_g
        wg = xg @ r_inv  # n_g x p
        u_g, s_g, _ = np.linalg.svd(wg, full_matrices=False)
        one_minus_s2 = 1.0 - s_g * s_g
        if float(np.min(one_minus_s2)) <= 1e-10:
            raise InsufficientSupportError(
                "CR2 adjustment matrix (I - H_gg) is singular for at least one cluster (cluster leverage = 1)"
            )
        boost = (one_minus_s2 ** -0.5) - 1.0
        u_factors.append((u_g, boost))
        x_blocks.append(xg)
        scores[j] = xg.T @ _apply_ag(u_g, boost, ug)

    meat = scores.T @ scores
    cov = _sandwich(bread, meat)

    # Satterthwaite effective degrees of freedom via exact identity:
    # P_{gh} = delta_{gh} ||v_g||^2 - (X_g' v_g)' B (X_h' v_h), where v_g = A_g X_g B c.
    c_mat = np.eye(p, dtype=float) if contrasts is None else np.atleast_2d(np.asarray(contrasts, dtype=float))
    if c_mat.shape[1] != p or not np.isfinite(c_mat).all():
        raise InvalidDesignError(f"contrasts must have {p} finite columns")
    dfs = np.empty(c_mat.shape[0], dtype=float)
    for k in range(c_mat.shape[0]):
        bc = bread @ c_mat[k]
        diag_norms = np.empty(g, dtype=float)
        u_mat = np.empty((g, p), dtype=float)
        for j in range(g):
            u_g, boost = u_factors[j]
            v_g = _apply_ag(u_g, boost, x_blocks[j] @ bc)
            diag_norms[j] = float(v_g @ v_g)
            u_mat[j] = x_blocks[j].T @ v_g
        p_mat = np.diag(diag_norms) - u_mat @ bread @ u_mat.T
        trace_p = float(np.trace(p_mat))
        trace_p2 = float(np.sum(p_mat * p_mat))
        dfs[k] = max(1.0, (trace_p * trace_p) / trace_p2) if trace_p2 > 1e-16 else 1.0

    return CR2Covariance(covariance=cov, n_clusters=g, satterthwaite_df=dfs)


@dataclass(frozen=True)
class MultiwayClusterCovariance:
    """Cameron-Gelbach-Miller (2011) multi-way cluster covariance with PSD audit."""

    covariance: np.ndarray
    cluster_counts: tuple[int, ...]
    min_eigenvalue_raw: float
    psd_adjusted: bool


def multiway_cluster_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    groups_list: Sequence[np.ndarray],
    *,
    small_sample: bool = True,
    enforce_psd: bool = True,
) -> MultiwayClusterCovariance:
    """Multi-way (2-way or higher) CGM (2011) cluster-robust covariance.

    Computes inclusion-exclusion over all nonempty intersections of cluster
    dimensions. Any negative eigenvalues from finite-sample inclusion-exclusion
    are explicitly recorded in ``min_eigenvalue_raw`` and ``psd_adjusted``.
    """
    if len(groups_list) < 2:
        raise InvalidDesignError("multiway_cluster_covariance requires at least two clustering dimensions")
    x, u, bread = _inputs(x, residuals)
    n, p = x.shape

    encoded = []
    counts = []
    for grp in groups_list:
        inv, g = _validate_groups(grp, n)
        encoded.append(inv)
        counts.append(g)

    total_cov = np.zeros((p, p), dtype=float)
    d_dims = len(encoded)
    for r in range(1, d_dims + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for subset in combinations(range(d_dims), r):
            if r == 1:
                inv = encoded[subset[0]]
                g = counts[subset[0]]
            else:
                # Deterministic tuple-key intersection encoding
                keys = np.column_stack([encoded[idx] for idx in subset])
                _, inv = np.unique(keys, axis=0, return_inverse=True)
                g = int(inv.max() + 1)
                if g < 2:
                    raise InsufficientSupportError("Cluster intersection has fewer than two distinct groups")
            scores = np.zeros((g, p), dtype=float)
            np.add.at(scores, inv, x * u[:, None])
            meat = scores.T @ scores
            if small_sample:
                meat *= (g / (g - 1)) * ((n - 1) / (n - p))
            total_cov += sign * _sandwich(bread, meat)

    total_cov = 0.5 * (total_cov + total_cov.T)
    eigvals, eigvecs = np.linalg.eigh(total_cov)
    min_eig = float(eigvals[0])
    psd_adjusted = False
    if enforce_psd and min_eig < 0.0:
        eigvals = np.maximum(eigvals, 0.0)
        total_cov = (eigvecs * eigvals) @ eigvecs.T
        total_cov = 0.5 * (total_cov + total_cov.T)
        psd_adjusted = True

    return MultiwayClusterCovariance(
        covariance=total_cov,
        cluster_counts=tuple(counts),
        min_eigenvalue_raw=min_eig,
        psd_adjusted=psd_adjusted,
    )
