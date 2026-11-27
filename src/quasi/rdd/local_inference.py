"""Experimental local-polynomial RD influence functions (not rdrobust parity).

All weights act on individual observations. The optional higher-order correction
uses a separate q=p+1 pilot fit on each side; its uncertainty is propagated
through the *same observation-level* influence vector as the point estimate.
This implements an HC0 residual plug-in, not rdrobust's NN/HC variance choices,
bandwidth optimization, mass-point adjustments, or coverage corrections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasi.rdd.kernels import KERNEL_REGISTRY


@dataclass(frozen=True)
class BoundaryWeights:
    conventional: np.ndarray  # maps outcome observations to cutoff intercept
    corrected: np.ndarray  # maps outcome observations to bias-corrected cutoff intercept
    active_h: np.ndarray
    active_b: np.ndarray
    left_or_right: str
    n_h: int
    n_b: int
    unique_h: int
    unique_b: int


def _projection(x: np.ndarray, bandwidth: float, degree: int, kernel: str,
                component: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = x / bandwidth
    w = KERNEL_REGISTRY[kernel](u)
    active = w > 0.0
    if int(active.sum()) < degree + 2 or np.unique(x[active]).size < degree + 2:
        raise ValueError("Local polynomial is unsupported: too few observations or distinct running values")
    Z = np.vander(u[active], N=degree + 1, increasing=True)
    weighted = Z * np.sqrt(w[active, None])
    if np.linalg.matrix_rank(weighted) < degree + 1:
        raise ValueError("Local polynomial design matrix is rank deficient")
    if np.linalg.cond(weighted) > 1e9:
        raise ValueError("Local polynomial is ill-conditioned; review running-variable support")
    # The projection row is equivalent to e_j'(Z'WZ)^-1 Z'W but is
    # calculated via a stable least-squares solve instead of inv(X'WX).
    coeff = np.linalg.pinv(weighted)
    projection = np.zeros(x.size, dtype=float)
    projection[active] = (coeff[component, :] * np.sqrt(w[active]))
    return projection, active, u


def boundary_weights(x: np.ndarray, h: float, b: float | None, degree: int,
                     kernel: str, side: str) -> BoundaryWeights:
    """Compute conventional and bias-corrected linear weights at one boundary."""
    mask = x >= 0 if side == "right" else x < 0
    xs = x[mask]
    if not len(xs):
        raise ValueError("RD needs running-variable observations on both sides of cutoff")
    p, active_h, _ = _projection(xs, h, degree, kernel, 0)
    corrected = p.copy()
    active_b = active_h
    if b is not None:
        q, active_b, _ = _projection(xs, b, degree + 1, kernel, degree + 1)
        # Local p intercept approximation to the omitted (p+1) polynomial:
        # bias_p = intercept_projection_h[x^(p+1)] * derivative_coeff_q.
        coefficient = np.dot(p, (xs / b) ** (degree + 1))
        corrected = p - coefficient * q
    full_p = np.zeros(x.size)
    full_corrected = np.zeros(x.size)
    full_p[mask] = p
    full_corrected[mask] = corrected
    hmask = np.zeros(x.size, dtype=bool)
    bmask = np.zeros(x.size, dtype=bool)
    hmask[mask] = active_h
    bmask[mask] = active_b
    return BoundaryWeights(full_p, full_corrected, hmask, bmask, side,
                           int(active_h.sum()), int(active_b.sum()),
                           int(np.unique(xs[active_h]).size),
                           int(np.unique(xs[active_b]).size))


def side_residuals(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    degree: int,
    kernel: str,
    side: str,
    *,
    eval_mask: np.ndarray | None = None,
) -> np.ndarray:
    """WLS residuals used for HC0 plug-in, calculated without dense W matrices.

    When ``eval_mask`` is supplied (e.g., union of h and b windows when b < h),
    the fitted side polynomial is evaluated at every active point in ``eval_mask``
    so no observation with a nonzero influence weight receives a silent zero residual.
    """
    side_mask = x >= 0 if side == "right" else x < 0
    u = x[side_mask] / bandwidth
    weights = KERNEL_REGISTRY[kernel](u)
    active = weights > 0
    design = np.vander(u[active], N=degree + 1, increasing=True)
    root_w = np.sqrt(weights[active])
    beta = np.linalg.lstsq(design * root_w[:, None], y[side_mask][active] * root_w, rcond=None)[0]
    residual = np.zeros(x.size)
    res_side = np.zeros(int(side_mask.sum()))
    target = active if eval_mask is None else (active | eval_mask[side_mask])
    design_eval = np.vander(u[target], N=degree + 1, increasing=True)
    res_side[target] = y[side_mask][target] - design_eval @ beta
    residual[side_mask] = res_side
    return residual


def _nn_residuals(x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, nn_matches: int = 3) -> np.ndarray:
    """CCT (2014) J-nearest-neighbor plug-in residuals on one cutoff side."""
    idx = np.flatnonzero(mask)
    if len(idx) <= nn_matches:
        raise ValueError(f"Insufficient observations ({len(idx)}) for J={nn_matches} nearest-neighbor RDD variance")
    xs = x[idx]
    ys = y[idx]
    order = np.argsort(xs, kind="mergesort")
    xs_s = xs[order]
    ys_s = ys[order]
    m = len(xs_s)
    res_s = np.empty(m, dtype=float)
    # For each observation i, average the J nearest distinct-index neighbors on the same side.
    for i in range(m):
        lo = max(0, i - nn_matches - 2)
        hi = min(m, i + nn_matches + 3)
        cand = [k for k in range(lo, hi) if k != i]
        cand.sort(key=lambda k: abs(xs_s[k] - xs_s[i]))
        nbrs = cand[:nn_matches]
        j_eff = len(nbrs)
        # Scale by sqrt(J / (J + 1)) so E[r_i^2] = sigma_i^2 (CCT 2014 Section 4.1)
        res_s[i] = np.sqrt(j_eff / (j_eff + 1.0)) * (ys_s[i] - float(np.mean(ys_s[nbrs])))
    out = np.zeros(x.size, dtype=float)
    out[idx[order]] = res_s
    return out


def rd_inference(
    x: np.ndarray,
    y: np.ndarray,
    h: float,
    kernel: str,
    degree: int,
    *,
    treatment: np.ndarray | None = None,
    bias_bandwidth: float | None = None,
    cluster: np.ndarray | None = None,
    vce: str = "hc0",
    nn_matches: int = 3,
) -> dict:
    """Sharp or fuzzy boundary jump with HC0/HC1/HC2/HC3/NN conventional or bias-corrected SE.

    A fuzzy local Wald ratio uses a joint, observation-level delta-method
    influence function: (IF_y - tau*IF_t)/first_stage, retaining covariance.
    """
    vce_l = vce.lower()
    if vce_l not in ("hc0", "hc1", "hc2", "hc3", "nn"):
        raise ValueError(f"Unsupported RDD vce={vce!r}; expected hc0, hc1, hc2, hc3, or nn")
    if treatment is not None and cluster is not None:
        raise NotImplementedError("Clustered fuzzy RD inference is not validated")
    if bias_bandwidth is not None and cluster is not None:
        raise NotImplementedError("Clustered bias-corrected RD inference is not validated")
    right = boundary_weights(x, h, bias_bandwidth, degree, kernel, 'right')
    left = boundary_weights(x, h, bias_bandwidth, degree, kernel, 'left')
    Lp = right.conventional - left.conventional
    L = right.corrected - left.corrected
    yjump = float(Lp @ y)
    yjump_bc = float(L @ y)
    trjump = 1.0
    trjump_bc = 1.0
    if treatment is not None:
        trjump = float(Lp @ treatment)
        trjump_bc = float(L @ treatment)
        # Wald ratios become unstable near a zero first-stage jump. This is
        # an explicit numerical guard, NOT a complete weak-IV diagnostic.
        if abs(trjump) <= 1e-5:
            raise ValueError("Fuzzy RD conventional first-stage discontinuity is near zero; local Wald ratio undefined")
    first_stage_se = None
    if treatment is not None:
        eps_t_conv = (side_residuals(x, treatment, h, degree, kernel, 'right') +
                      side_residuals(x, treatment, h, degree, kernel, 'left'))
        first_stage_se = float(np.sqrt(np.sum((Lp * eps_t_conv)**2)))
    tau = yjump / trjump
    if treatment is not None and bias_bandwidth is not None:
        # rdrobust's fuzzy RBC uses a *linearized* bias correction evaluated
        # at the conventional Wald ratio, NOT the ratio of two corrected jumps.
        # tau_bc = tau - grad_g(yjump, trjump) @ estimated_jump_biases.
        # This distinction matters when the first-stage has nonzero curvature.
        tau_bc = tau - ((yjump - yjump_bc) / trjump
                        - yjump * (trjump - trjump_bc) / trjump**2)
    else:
        tau_bc = yjump_bc / trjump_bc
    rdegree = degree + (bias_bandwidth is not None)
    rbandwidth = bias_bandwidth if bias_bandwidth is not None else h
    union_right = right.active_h | right.active_b
    union_left = left.active_h | left.active_b

    def _get_residuals(vec: np.ndarray) -> np.ndarray:
        if cluster is None and vce_l == "nn":
            return _nn_residuals(x, vec, union_right, nn_matches=nn_matches) + _nn_residuals(
                x, vec, union_left, nn_matches=nn_matches
            )
        raw_res = (
            side_residuals(x, vec, rbandwidth, rdegree, kernel, "right", eval_mask=union_right)
            + side_residuals(x, vec, rbandwidth, rdegree, kernel, "left", eval_mask=union_left)
        )
        if cluster is not None or vce_l == "hc0":
            return raw_res
        # Finite-sample HC1/HC2/HC3 leverage adjustment strictly on active residual-fit observations per side
        adj = raw_res.copy()
        active_r = right.active_b if bias_bandwidth is not None else right.active_h
        active_l = left.active_b if bias_bandwidth is not None else left.active_h
        for smask in (active_r, active_l):
            idx = np.flatnonzero(smask)
            n_s = len(idx)
            p_s = rdegree + 1
            if n_s <= p_s:
                continue
            if vce_l == "hc1":
                adj[idx] *= np.sqrt(n_s / (n_s - p_s))
            else:
                u_s = x[idx] / rbandwidth
                w_s = KERNEL_REGISTRY[kernel](u_s)
                z_s = np.vander(u_s, N=p_s, increasing=True)
                zw = z_s * np.sqrt(w_s)[:, None]
                q_s, _ = np.linalg.qr(zw, mode="reduced")
                h_diag = np.clip(np.sum(q_s * q_s, axis=1), 0.0, 0.999)
                power = 0.5 if vce_l == "hc2" else 1.0
                adj[idx] /= (1.0 - h_diag) ** power
        return adj

    eps_y = _get_residuals(y)
    if treatment is None:
        score = L * eps_y
    else:
        eps_t = _get_residuals(treatment)
        # The bias-corrected fuzzy influence function is based on the
        # conventional Wald linearization point, including its denominator.
        # The conventional branch also uses tau and trjump, identically.
        score = L * (eps_y - tau * eps_t) / trjump
    if cluster is None:
        var = float(score @ score)
    else:
        in_support = left.active_h | right.active_h
        ids, inv = np.unique(cluster[in_support], return_inverse=True)
        if (len(np.unique(cluster[left.active_h])) < 2 or
            len(np.unique(cluster[right.active_h])) < 2):
            raise ValueError("Clustered RD requires at least two independent clusters per cutoff side")
        if len(ids) < 10:
            raise ValueError("Conventional clustered RD requires at least 10 independent clusters")
        cluster_score = np.bincount(inv, weights=score[in_support], minlength=len(ids))
        var = float(cluster_score @ cluster_score) * len(ids) / (len(ids) - 1)
    if not np.isfinite(var) or var <= 0:
        raise ValueError("RD covariance nonpositive or nonfinite; inference is undefined")
    return {
        'estimate': float(tau_bc if bias_bandwidth is not None else tau),
        'se': float(np.sqrt(var)),
        'conventional_estimate': float(tau),
        'bias_corrected_estimate': float(tau_bc) if bias_bandwidth is not None else None,
        'first_stage_jump': trjump if treatment is not None else None,
        'first_stage_se': first_stage_se,
        'bias_corrected_first_stage_jump': trjump_bc if treatment is not None and bias_bandwidth is not None else None,
        'fuzzy_bias_correction': ('conventional_wald_first_order' if treatment is not None and bias_bandwidth is not None else None),
        'n_left_h': left.n_h, 'n_right_h': right.n_h,
        'n_left_b': left.n_b, 'n_right_b': right.n_b,
        'n_unique_left': left.unique_h, 'n_unique_right': right.unique_h,
        'inference_method': (f'experimental_bias_corrected_{vce_l}' if bias_bandwidth is not None
                             else ('experimental_cluster_cr1' if cluster is not None else f'conventional_{vce_l}')),
    }


def fuzzy_rd_ar_test(
    x: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    h: float,
    kernel: str,
    degree: int,
    null_effect: float = 0.0,
    *,
    bias_bandwidth: float | None = None,
    vce: str = "hc0",
) -> dict:
    """Feir, Lemieux & Marmer (2016) weak-identification-robust local AR test for Fuzzy RD.

    Tests H0: tau = null_effect by testing zero discontinuity in (Y - null_effect * T)
    at the cutoff, completely bypassing division by a potentially weak first stage.
    """
    if isinstance(null_effect, bool) or not np.isscalar(null_effect) or not np.isfinite(null_effect):
        raise ValueError("null_effect must be a finite scalar")
    from scipy import stats

    reduced = np.asarray(y, dtype=float) - float(null_effect) * np.asarray(treatment, dtype=float)
    fit = rd_inference(x, reduced, h, kernel, degree, bias_bandwidth=bias_bandwidth, vce=vce)
    z_stat = float(fit["estimate"] / fit["se"])
    p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
    return {
        "null_effect": float(null_effect),
        "statistic": z_stat * z_stat,
        "z_statistic": z_stat,
        "p_value": p_val,
        "distribution": "chi2(1)",
        "method": f"fuzzy_rd_anderson_rubin_{vce.lower()}",
    }
