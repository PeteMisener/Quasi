"""Experimental AR(1)-error feasible GLS (Prais--Winsten transformation).

This estimates correlation from residuals; covariance and t inference condition
on the estimated rho. It is not a guaranteed finite-sample correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from quasi.errors import InsufficientSupportError, InvalidDesignError, NotIdentifiedError
from quasi.inference.linear import least_squares


@dataclass(frozen=True)
class PraisWinstenFit:
    coefficients: np.ndarray
    covariance: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rho: float
    iterations: int
    degrees_freedom: int
    innovation_variance: float


def prais_winsten_ar1(
    x: np.ndarray, y: np.ndarray, *, max_iter: int = 100,
    tolerance: float = 1e-7,
) -> PraisWinstenFit:
    """Iterative PW FGLS retaining t=0 under a stationary AR(1) error model.

    Fail closed on near-unit-root rho or non-convergence. The estimated
    rho is treated as fixed in the reported coefficient covariance.
    """
    if type(max_iter) is not int or max_iter < 2 or max_iter > 10000:
        raise InvalidDesignError("ar1_max_iter must be an integer between 2 and 10000")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or not 0 < tolerance < 0.1:
        raise InvalidDesignError("ar1_tolerance must be finite and between zero and 0.1")
    z = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    initial = least_squares(z, yy)
    n, p = z.shape
    if n <= p + 4:
        raise InsufficientSupportError("Prais-Winsten AR(1) requires more than p+4 periods")

    def estimate_rho(resid: np.ndarray) -> float:
        denom = float(np.dot(resid[:-1], resid[:-1]))
        if not np.isfinite(denom) or denom <= 1e-14:
            raise InsufficientSupportError("AR(1) correlation is not identified with negligible residual variation")
        r = float(np.dot(resid[1:], resid[:-1]) / denom)
        if not np.isfinite(r) or abs(r) >= 0.98:
            raise InsufficientSupportError("Estimated AR(1) correlation is near or beyond the stationary boundary; use a different error specification")
        return r

    def whiten(r: float) -> tuple[np.ndarray, np.ndarray]:
        xx = z.copy()
        target = yy.copy()
        xx[0] *= np.sqrt(1 - r*r)
        target[0] *= np.sqrt(1 - r*r)
        xx[1:] = z[1:] - r*z[:-1]
        target[1:] = yy[1:] - r*yy[:-1]
        return xx, target

    rho = estimate_rho(initial.residuals)
    for iteration in range(1, max_iter + 1):
        xw, yw = whiten(rho)
        fitted = least_squares(xw, yw)
        new_resid = yy - z @ fitted.coefficients
        updated = estimate_rho(new_resid)
        if abs(updated - rho) <= tolerance:
            rho = updated
            break
        rho = updated
    else:
        raise InsufficientSupportError("Prais-Winsten AR(1) iteration failed to converge")
    # Fit again at final rho, so all estimates and covariance use the same transformation.
    xw, yw = whiten(rho)
    fit = least_squares(xw, yw)
    resid_w = yw - xw @ fit.coefficients
    df = n-p
    sigma2 = float(resid_w @ resid_w / df)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise InsufficientSupportError("Nonpositive Prais-Winsten innovation variance")
    _, sing, vt = np.linalg.svd(xw, full_matrices=False)
    covariance = sigma2 * ((vt.T / (sing*sing)) @ vt)
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
        raise InsufficientSupportError("Invalid Prais-Winsten covariance")
    return PraisWinstenFit(fit.coefficients, covariance, z @ fit.coefficients,
                          yy-z @ fit.coefficients, rho, iteration, df, sigma2)


def profile_ml_ar1(
    x: np.ndarray, y: np.ndarray, *, tolerance: float = 1e-7,
) -> PraisWinstenFit:
    """Stationary Gaussian AR(1) *profile maximum-likelihood* regression.

    For each rho, profile out beta and innovation variance using the exact
    stationary first-observation likelihood. The returned covariance is the
    conditional-on-rho GLS covariance using SSE/(n-p), *not* a joint Hessian
    covariance or an automatically calibrated small-sample interval.

    Restrictions: one Gaussian AR(1) error process and regular time cadence.
    Rho optimization is bounded away from the unit-root boundary; apparent
    boundary solutions fail instead of yielding reassuring inference.
    """
    from scipy.optimize import minimize_scalar

    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or not np.isfinite(tolerance) or not 0 < tolerance < 0.1:
        raise InvalidDesignError("ar1_tolerance must be finite and between zero and 0.1")
    z = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if z.ndim != 2 or yy.ndim != 1 or z.shape[0] != len(yy):
        raise InvalidDesignError("Expected an n-by-p design matrix and matching one-dimensional outcome")
    n, p = z.shape
    if n <= p+4:
        raise InsufficientSupportError("AR(1) profile likelihood requires more than p+4 periods")
    # Shared routine establishes full rank and finite input first.
    least_squares(z, yy)

    def transform(rho: float) -> tuple[np.ndarray, np.ndarray]:
        weight = np.sqrt(1-rho*rho)
        xw = np.empty_like(z)
        yw = np.empty_like(yy)
        xw[0] = weight*z[0]
        yw[0] = weight*yy[0]
        xw[1:] = z[1:] - rho*z[:-1]
        yw[1:] = yy[1:] - rho*yy[:-1]
        return xw, yw

    def objective(rho: float) -> float:
        xw, yw = transform(rho)
        try:
            resid = least_squares(xw, yw).residuals
            sse = float(resid @ resid)
        except InsufficientSupportError:
            return float("inf")
        if not np.isfinite(sse) or sse <= 1e-14:
            return float("inf")
        # Exact stationary Gaussian AR1 likelihood, concentrated over beta
        # and sigma2_annual innovations: -2*logL = n*log(SSE/n) - log(1-rho^2) + const.
        return 0.5*n*np.log(sse/n) - 0.5*np.log1p(-rho*rho)

    # A bounded 1D search is not universally globally convex. Checking a
    # coarse grid and refining its most promising subinterval reduces the
    # chance of accepting a local solution under adverse small samples.
    grid = np.linspace(-0.975, 0.975, 79)
    values = np.asarray([objective(float(r)) for r in grid])
    if not np.isfinite(values).any():
        raise InsufficientSupportError("AR(1) likelihood is not finite on the search grid")
    i = int(np.argmin(values))
    if i in (0, len(grid)-1):
        raise InsufficientSupportError("AR(1) likelihood optimum is near the stationary boundary")
    sol = minimize_scalar(objective, bounds=(float(grid[i-1]), float(grid[i+1])),
                          method="bounded", options={"xatol": min(float(tolerance), 1e-8)})
    if not sol.success or not np.isfinite(sol.fun):
        raise InsufficientSupportError("AR(1) profile likelihood optimizer failed")
    rho = float(sol.x)
    if abs(rho) >= 0.97:
        raise InsufficientSupportError("AR(1) likelihood estimate is near stationary boundary")
    xw, yw = transform(rho)
    fit = least_squares(xw, yw)
    residual = yw - xw @ fit.coefficients
    innovation_mle = float(residual @ residual / n)
    sigma2_df = float(residual @ residual / (n-p))
    if innovation_mle <= 0 or not np.isfinite(innovation_mle):
        raise InsufficientSupportError("Nonpositive AR(1) maximum-likelihood innovation variance")
    _, s, vt = np.linalg.svd(xw, full_matrices=False)
    conditional_cov = sigma2_df*((vt.T/(s*s))@vt)
    if not np.isfinite(conditional_cov).all() or np.any(np.diag(conditional_cov) <= 0):
        raise InsufficientSupportError("Invalid AR(1) conditional coefficient covariance")
    return PraisWinstenFit(
        fit.coefficients, conditional_cov, z @ fit.coefficients,
        yy-z @ fit.coefficients, rho, int(sol.nit), n-p, innovation_mle,
    )


def reml_ar1(
    x: np.ndarray, y: np.ndarray, *, tolerance: float = 1e-7,
) -> PraisWinstenFit:
    """Stationary Gaussian AR(1) Restricted Maximum Likelihood (REML).

    Adds the Harville (1974) +0.5 * log|X_w(rho)' X_w(rho)| projection term to
    the profile log-likelihood, reducing finite-sample downward (Hurwicz) bias
    in estimated rho when estimating level, slope, and seasonal parameters.
    """
    from scipy.optimize import minimize_scalar

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or not 0 < tolerance < 0.1
    ):
        raise InvalidDesignError("ar1_tolerance must be finite and between zero and 0.1")
    z = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if z.ndim != 2 or yy.ndim != 1 or z.shape[0] != len(yy):
        raise InvalidDesignError("Expected an n-by-p design matrix and matching one-dimensional outcome")
    n, p = z.shape
    if n <= p + 4:
        raise InsufficientSupportError("AR(1) REML requires more than p+4 periods")
    least_squares(z, yy)

    def transform(rho: float) -> tuple[np.ndarray, np.ndarray]:
        weight = np.sqrt(1.0 - rho * rho)
        xw = np.empty_like(z)
        yw = np.empty_like(yy)
        xw[0] = weight * z[0]
        yw[0] = weight * yy[0]
        xw[1:] = z[1:] - rho * z[:-1]
        yw[1:] = yy[1:] - rho * yy[:-1]
        return xw, yw

    def objective(rho: float) -> float:
        xw, yw = transform(rho)
        try:
            fit = least_squares(xw, yw)
            sse = float(fit.residuals @ fit.residuals)
        except (NotIdentifiedError, InsufficientSupportError):
            return float("inf")
        if not np.isfinite(sse) or sse <= 1e-14:
            return float("inf")
        sing = np.linalg.svd(xw, compute_uv=False)
        if sing[-1] <= 1e-14:
            return float("inf")
        log_det_half = float(np.sum(np.log(sing)))
        return 0.5 * (n - p) * np.log(sse / (n - p)) - 0.5 * np.log1p(-rho * rho) + log_det_half

    grid = np.linspace(-0.975, 0.975, 79)
    values = np.asarray([objective(float(r)) for r in grid])
    if not np.isfinite(values).any():
        raise InsufficientSupportError("AR(1) REML objective is not finite on the search grid")
    i = int(np.argmin(values))
    if i in (0, len(grid) - 1):
        raise InsufficientSupportError("AR(1) REML optimum is near the stationary boundary")
    sol = minimize_scalar(
        objective,
        bounds=(float(grid[i - 1]), float(grid[i + 1])),
        method="bounded",
        options={"xatol": min(float(tolerance), 1e-8)},
    )
    if not sol.success or not np.isfinite(sol.fun):
        raise InsufficientSupportError("AR(1) REML optimizer failed")
    rho = float(sol.x)
    if abs(rho) >= 0.97:
        raise InsufficientSupportError("AR(1) REML estimate is near stationary boundary")
    xw, yw = transform(rho)
    fit = least_squares(xw, yw)
    residual = yw - xw @ fit.coefficients
    sigma2_reml = float(residual @ residual / (n - p))
    if sigma2_reml <= 0 or not np.isfinite(sigma2_reml):
        raise InsufficientSupportError("Nonpositive AR(1) REML innovation variance")
    _, s, vt = np.linalg.svd(xw, full_matrices=False)
    conditional_cov = sigma2_reml * ((vt.T / (s * s)) @ vt)
    if not np.isfinite(conditional_cov).all() or np.any(np.diag(conditional_cov) <= 0):
        raise InsufficientSupportError("Invalid AR(1) REML conditional coefficient covariance")
    return PraisWinstenFit(
        fit.coefficients,
        conditional_cov,
        z @ fit.coefficients,
        yy - z @ fit.coefficients,
        rho,
        int(sol.nit),
        n - p,
        sigma2_reml,
    )
