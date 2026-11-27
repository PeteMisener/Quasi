"""Checked convex simplex-QP and simplex-regression solvers for SCM and SDiD.

Only convex QP with symmetric PSD Q is supported. Every solution is checked for
simplex feasibility and KKT projected-gradient stationarity; unconverged fits
fail closed with ConvergenceError.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quasi.errors import ConvergenceError, InvalidDesignError


def project_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {w >= 0 : sum(w) = s} in O(N log N) time."""
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = arr.shape[0]
    if n == 0 or not np.isfinite(arr).all() or not np.isfinite(s) or s <= 0:
        raise InvalidDesignError("Simplex projection requires finite vector and positive radius")
    u = np.sort(arr)[::-1]
    cssv = np.cumsum(u) - s
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, s / n, dtype=float)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(arr - theta, 0.0)


@dataclass(frozen=True)
class SimplexSolution:
    weights: np.ndarray
    objective: float
    iterations: int
    projected_stationarity: float
    converged: bool
    intercept: float = 0.0


def checked_simplex_qp(
    q: np.ndarray,
    p: np.ndarray,
    *,
    max_iter: int = 10000,
    tol: float = 1e-8,
) -> SimplexSolution:
    """Solve min 0.5 w'Q w - p'w over simplex via monotone restarted FISTA."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[0] != q.shape[1] or p.shape != (q.shape[0],):
        raise InvalidDesignError("QP requires square Q and conformable vector p")
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise InvalidDesignError("QP Q/p must be finite")
    if type(max_iter) is not int or max_iter < 1 or not np.isfinite(tol) or not 0 < tol < 1:
        raise InvalidDesignError("max_iter must be positive int and tolerance in (0,1)")
    if not np.allclose(q, q.T, atol=1e-10, rtol=1e-10):
        raise InvalidDesignError("QP requires symmetric Q")
    q = (q + q.T) * 0.5
    eigen = np.linalg.eigvalsh(q)
    if eigen[0] < -1e-10 * max(1.0, abs(float(eigen[-1]))):
        raise InvalidDesignError("QP objective must be convex (Q PSD)")
    lipschitz = max(float(eigen[-1]), 1e-12)
    step = 1.0 / lipschitz
    n = len(p)
    x = np.ones(n, dtype=float) / n
    y = x.copy()
    t = 1.0

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ q @ w) - float(p @ w)

    for iteration in range(1, max_iter + 1):
        candidate = project_simplex(y - step * (q @ y - p))
        if objective(candidate) > objective(x) + 1e-12:
            # Monotone FISTA restart: safeguard on extrapolation overshoot.
            t = 1.0
            y = x
            candidate = project_simplex(y - step * (q @ y - p))
        residual = float(np.max(np.abs(candidate - project_simplex(candidate - step * (q @ candidate - p)))))
        if residual <= tol:
            return SimplexSolution(candidate, objective(candidate), iteration, residual, True)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = candidate + ((t - 1.0) / t_next) * (candidate - x)
        x, t = candidate, t_next
    raise ConvergenceError(
        f"Convex simplex QP did not satisfy stationarity tolerance {tol} in {max_iter} iterations"
    )


def solve_simplex_regression(
    A: np.ndarray,
    b: np.ndarray,
    *,
    ridge: float = 0.0,
    intercept: bool = False,
    kkt_tol: float = 1e-5,
    max_iter: int = 20000,
) -> SimplexSolution:
    """Minimize (1/(2m))||b - a - A w||^2 + 0.5 * ridge^2 ||w||^2 over the unit simplex.

    Profiles out an unpenalized intercept ``a`` when ``intercept=True``. Uses SLSQP
    with analytic Jacobian and verifies feasibility + KKT stationarity, falling back
    to monotone restarted FISTA if SLSQP stalls before stationarity.
    """
    A_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float).reshape(-1)
    if A_arr.ndim != 2 or b_arr.shape != (A_arr.shape[0],) or A_arr.shape[0] < 2 or A_arr.shape[1] < 2:
        raise InvalidDesignError("Simplex regression requires at least two observations and two weights")
    if not np.isfinite(A_arr).all() or not np.isfinite(b_arr).all() or not np.isfinite(ridge) or ridge < 0:
        raise InvalidDesignError("Simplex regression requires finite inputs and nonnegative ridge")

    if intercept:
        a_bar = A_arr.mean(axis=0)
        b_bar = float(b_arr.mean())
        X = A_arr - a_bar
        y = b_arr - b_bar
    else:
        # On the simplex sum(w) == 1, subtracting a common scalar c from both A and b
        # leaves A @ w - b identically unchanged while preserving conditioning under huge shifts.
        common_shift = float(b_arr.mean())
        a_bar = np.zeros(A_arr.shape[1], dtype=float)
        b_bar = 0.0
        X = A_arr - common_shift
        y = b_arr - common_shift

    m, n = X.shape
    # Normalize internally by problem scale so SLSQP/FISTA tolerances are strictly
    # invariant to arbitrary rescaling of outcomes (Y -> s * Y), including small-scale rates.
    problem_scale = max(
        float(np.linalg.norm(X, ord="fro")) / np.sqrt(m),
        float(np.linalg.norm(y)) / np.sqrt(m),
        float(ridge),
    )
    if problem_scale <= 1e-14:
        w0 = np.full(n, 1.0 / n, dtype=float)
        return SimplexSolution(
            weights=w0,
            objective=0.0,
            iterations=0,
            projected_stationarity=0.0,
            converged=True,
            intercept=float(b_bar - a_bar @ w0),
        )
    Xs = X / problem_scale
    ys = y / problem_scale
    rs = ridge / problem_scale

    Q = (Xs.T @ Xs) / m + (rs**2) * np.eye(n)
    Q = 0.5 * (Q + Q.T)
    p = (Xs.T @ ys) / m

    def fun(w: np.ndarray) -> float:
        r = Xs @ w - ys
        return float((r @ r) / (2.0 * m) + 0.5 * (rs**2) * (w @ w))

    def jac(w: np.ndarray) -> np.ndarray:
        return Q @ w - p

    scale = max(float(np.linalg.norm(Q, ord=2)), float(np.linalg.norm(p)), 1e-12)

    def kkt_residual(w: np.ndarray) -> float:
        return float(np.max(np.abs(w - project_simplex(w - jac(w) / scale))))

    sol = minimize(
        fun,
        np.full(n, 1.0 / n, dtype=float),
        method="SLSQP",
        jac=jac,
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones(n)}],
        options={"maxiter": max_iter, "ftol": 1e-14},
    )
    w = np.asarray(sol.x, dtype=float)
    res = kkt_residual(w) if np.isfinite(w).all() else float("inf")
    if sol.success and np.isfinite(w).all() and w.min() >= -1e-8 and abs(w.sum() - 1.0) <= 1e-8 and res <= kkt_tol:
        w = np.maximum(w, 0.0)
        w /= w.sum()
        return SimplexSolution(
            weights=w,
            objective=fun(w) * (problem_scale**2),
            iterations=int(getattr(sol, "nit", 0)),
            projected_stationarity=res,
            converged=True,
            intercept=float(b_bar - a_bar @ w),
        )

    # Fallback: monotone restarted FISTA on the same convex objective, checked to kkt_tol.
    fista = checked_simplex_qp(Q, p, max_iter=max_iter, tol=min(kkt_tol, 1e-7))
    w_f = np.maximum(fista.weights, 0.0)
    w_f /= w_f.sum()
    res_f = kkt_residual(w_f)
    if not np.isfinite(w_f).all() or res_f > kkt_tol:
        raise ConvergenceError(
            f"Simplex optimization failed feasibility/stationarity: {sol.message}; KKT residual={res_f:.4g}"
        )
    return SimplexSolution(
        weights=w_f,
        objective=fun(w_f) * (problem_scale**2),
        iterations=fista.iterations,
        projected_stationarity=res_f,
        converged=True,
        intercept=float(b_bar - a_bar @ w_f),
    )
