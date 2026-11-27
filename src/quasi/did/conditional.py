"""Experimental traditional panel DR DiD ATT and nuisance-aware influence functions.

Implements the algebra of the *traditional* (not improved/IPT) estimator in
Sant'Anna & Zhao (2020), as documented in R DRDID::drdid_panel. This code is
NOT independently R-parity-certified. All checks intentionally fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class DRCell:
    att: float
    influence: np.ndarray  # length n of eligible two-group sample; sum ~0
    min_propensity: float
    max_propensity: float


def _invert_information(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e10:
        raise ValueError(f"DR DiD {label} singular/ill-conditioned; reduce covariates")
    return np.linalg.solve(matrix, np.eye(matrix.shape[0]))


def traditional_dr_panel(delta: np.ndarray, treated: np.ndarray, covariates: np.ndarray,
                         overlap_limit: float = .01) -> DRCell:
    """Traditional two-period panel DR ATT, conditioning on eligible sample.

    Parameters
    ----------
    delta: observed outcome change (post minus base), length n
    treated: boolean membership in the focal adoption cohort, length n
    covariates: baseline-period covariates, n by k *without* intercept
    overlap_limit: fail rather than truncate if any fitted propensity lies outside
        [limit, 1-limit]. Does not drop units or change the estimand silently.

    Returns estimate and first-order analytical influence function accounting
    for *both* logit propensity and OLS control outcome nuisance estimation.
    """
    dy = np.asarray(delta, dtype=float).reshape(-1)
    d = np.asarray(treated, dtype=float).reshape(-1)
    x = np.asarray(covariates, dtype=float)
    if x.ndim != 2 or x.shape[0] != dy.size or d.shape != dy.shape or x.shape[1] == 0:
        raise ValueError("DR DiD needs a nonempty 2D baseline covariate matrix with matching rows")
    if not np.isfinite(dy).all() or not np.isfinite(x).all() or not np.isin(d, (0, 1)).all():
        raise ValueError("DR DiD requires finite outcomes/covariates and binary treated group")
    if not isinstance(overlap_limit, (int, float)) or not 0 < overlap_limit < .5:
        raise ValueError("overlap_limit must lie between 0 and .5")
    n = dy.size
    n_t = int(d.sum())
    n_c = n - n_t
    k = x.shape[1] + 1
    if min(n_t, n_c) <= k or n <= k + 2:
        raise ValueError("DR DiD needs more treated and control units than regression parameters")
    X = np.column_stack((np.ones(n), x))
    if np.linalg.matrix_rank(X) < k or np.linalg.matrix_rank(X[d == 0]) < k:
        raise ValueError("DR DiD covariates are rank-deficient in the sample or controls")

    # Logistic MLE with analytic score and Hessian. Unlike naive model fitting,
    # complete/quasi separation and nonconvergence produce explicit errors.
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        p = expit(eta)
        return float(np.logaddexp(0, eta).sum() - d @ eta), X.T @ (p - d)

    initial = np.zeros(k)
    initial[0] = np.log(n_t / n_c)
    optimizer = minimize(objective, initial, method="BFGS", jac=True,
                         options={"maxiter": 500, "gtol": 1e-7})
    beta = optimizer.x
    score_norm = float(np.max(np.abs(objective(beta)[1])))
    if not np.isfinite(beta).all() or (not optimizer.success and score_norm > 1e-5) or score_norm > 1e-4:
        raise ValueError("DR DiD propensity logit failed to converge; possible separation")
    ps = expit(X @ beta)
    if np.min(ps) < overlap_limit or np.max(ps) > 1 - overlap_limit:
        raise ValueError("DR DiD propensity overlap violation; estimate not reported")

    # Outcome changes modeled on controls only, using unweighted OLS.
    control = (1 - d)
    Xc = X[d == 0]
    yc = dy[d == 0]
    outcome_coef = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    out = X @ outcome_coef
    residual = dy - out

    wt = d
    wc = control * ps / (1 - ps)
    denom_t = wt.mean()
    denom_c = wc.mean()
    eta_t = float(np.mean(wt * residual) / denom_t)
    eta_c = float(np.mean(wc * residual) / denom_c)
    att = eta_t - eta_c

    # Explicit nuisance IFs. The control OLS and propensity-logit estimation
    # terms are essential for calibrated asymptotic SEs; a residual-only IF is
    # generally not correct for the traditional DR estimator.
    info_out = (X.T @ (control[:, None] * X)) / n
    info_ps = (X.T @ ((ps * (1-ps))[:, None] * X)) / n
    asy_out = (control * residual)[:, None] * X @ _invert_information(info_out, "outcome information")
    asy_ps = ((d - ps)[:, None] * X) @ _invert_information(info_ps, "propensity information")
    M1 = np.mean(wt[:, None] * X, axis=0)
    M2 = np.mean((wc * (residual - eta_c))[:, None] * X, axis=0)
    M3 = np.mean(wc[:, None] * X, axis=0)
    inf_t = (wt * (residual - eta_t) - asy_out @ M1) / denom_t
    inf_c = (wc * (residual - eta_c) + asy_ps @ M2 - asy_out @ M3) / denom_c
    psi = inf_t - inf_c
    if not np.isfinite(psi).all():
        raise ValueError("DR DiD nuisance influence function is not finite")
    return DRCell(att=float(att), influence=psi, min_propensity=float(ps.min()), max_propensity=float(ps.max()))


def improved_dr_panel(
    delta: np.ndarray,
    treated: np.ndarray,
    covariates: np.ndarray,
    overlap_limit: float = 0.01,
) -> DRCell:
    """Sant'Anna & Zhao (2020) Improved Locally Efficient DR Panel DiD (IPT + WLS).

    Uses Inverse Probability Tilting (IPT) for the propensity score so that
    weighted control covariate means match treated covariate means exactly, paired
    with IPT-weighted control outcome regression. Because both nuisance first-order
    conditions orthogonalize the ATT moment, the influence function achieves
    doubly robust inference even if one nuisance model is misspecified.
    """
    dy = np.asarray(delta, dtype=float).reshape(-1)
    d = np.asarray(treated, dtype=float).reshape(-1)
    x = np.asarray(covariates, dtype=float)
    if x.ndim != 2 or x.shape[0] != dy.size or d.shape != dy.shape or x.shape[1] == 0:
        raise ValueError("Improved DR DiD needs a nonempty 2D baseline covariate matrix")
    if not np.isfinite(dy).all() or not np.isfinite(x).all() or not np.isin(d, (0, 1)).all():
        raise ValueError("Improved DR DiD requires finite outcomes/covariates and binary treated group")
    if not isinstance(overlap_limit, (int, float)) or not 0 < overlap_limit < 0.5:
        raise ValueError("overlap_limit must lie between 0 and 0.5")
    n = dy.size
    n_t = int(d.sum())
    n_c = n - n_t
    k = x.shape[1] + 1
    if min(n_t, n_c) <= k or n <= k + 2:
        raise ValueError("Improved DR DiD needs more treated and control units than parameters")
    # Standardize covariates internally for numerically stable IPT optimization
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    if np.any(scale < 1e-12) or not np.isfinite(scale).all():
        raise ValueError("Improved DR DiD covariates contain constant or degenerate columns")
    xs = np.column_stack((np.ones(n), (x - center) / scale))
    if np.linalg.matrix_rank(xs) < k or np.linalg.matrix_rank(xs[d == 0]) < k:
        raise ValueError("Improved DR DiD covariates are rank-deficient")

    # IPT objective (strictly convex): min_gamma mean((1 - d) * exp(X gamma) - d * X gamma)
    def ipt_obj(gamma: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(xs @ gamma, -30.0, 30.0)
        exp_eta = np.exp(eta)
        loss = float(np.mean((1.0 - d) * exp_eta - d * eta))
        grad = (xs.T @ ((1.0 - d) * exp_eta - d)) / n
        return loss, grad

    def ipt_hess(gamma: np.ndarray) -> np.ndarray:
        eta = np.clip(xs @ gamma, -30.0, 30.0)
        w = (1.0 - d) * np.exp(eta)
        return (xs.T * w) @ xs / n

    init = np.zeros(k)
    init[0] = np.log(n_t / n_c)
    opt = minimize(ipt_obj, init, method="trust-ncg", jac=True, hess=ipt_hess, options={"gtol": 1e-10, "maxiter": 500})
    gamma = opt.x
    grad_norm = float(np.max(np.abs(ipt_obj(gamma)[1])))
    if not np.isfinite(gamma).all() or grad_norm > 1e-5:
        opt2 = minimize(ipt_obj, init, method="BFGS", jac=True, options={"gtol": 1e-9, "maxiter": 1000})
        if np.isfinite(opt2.x).all() and float(np.max(np.abs(ipt_obj(opt2.x)[1]))) < grad_norm:
            gamma = opt2.x
            grad_norm = float(np.max(np.abs(ipt_obj(gamma)[1])))
    if not np.isfinite(gamma).all() or grad_norm > 1e-4:
        raise ValueError("Improved DR DiD IPT propensity optimization failed to converge")

    eta = xs @ gamma
    ps = expit(eta)
    if np.min(ps) < overlap_limit or np.max(ps) > 1.0 - overlap_limit:
        raise ValueError("Improved DR DiD propensity overlap violation; estimate not reported")

    odds = np.exp(eta)
    w_c = (1.0 - d) * odds
    # IPT-weighted OLS on controls (Sant'Anna & Zhao 2020 eq. 2.8)
    ctrl_mask = d == 0
    sw = np.sqrt(odds[ctrl_mask])
    xc_w = xs[ctrl_mask] * sw[:, None]
    yc_w = dy[ctrl_mask] * sw
    beta, _, rank, _ = np.linalg.lstsq(xc_w, yc_w, rcond=None)
    if rank < k:
        raise ValueError("Improved DR DiD weighted control regression is rank-deficient")

    residual = dy - xs @ beta
    p_t = float(d.mean())
    w_t = d / p_t
    w_ctrl_norm = w_c / float(w_c.mean())
    att = float(np.mean((w_t - w_ctrl_norm) * residual))
    # Under IPT + WLS, first-order nuisance derivative terms cancel identically.
    psi = w_t * (residual - att) - w_ctrl_norm * residual
    if not np.isfinite(psi).all():
        raise ValueError("Improved DR DiD influence function is not finite")
    return DRCell(att=att, influence=psi, min_propensity=float(ps.min()), max_propensity=float(ps.max()))
