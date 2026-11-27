"""Rank-aware linear solvers (unweighted & weighted) without N x N projectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quasi.errors import InvalidDesignError, NotIdentifiedError


@dataclass(frozen=True)
class LinearFit:
    coefficients: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    rank: int
    condition_number: float
    n_obs: int
    n_parameters: int


def _design(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise InvalidDesignError(f"{name} must be a nonempty, finite 2D matrix")
    if a.shape[0] <= a.shape[1]:
        raise NotIdentifiedError(f"{name} must have more observations than columns")
    return a


def _outcome(y: np.ndarray, n: int) -> np.ndarray:
    yy = np.asarray(y, dtype=np.float64)
    if yy.shape != (n,) or not np.isfinite(yy).all():
        raise InvalidDesignError("outcome must be a finite vector aligned with the design")
    return yy


def _weights(weights: np.ndarray | None, n: int) -> np.ndarray | None:
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w <= 0.0):
        raise InvalidDesignError("weights must be finite strictly positive numbers aligned with rows")
    return w


def _solve(a: np.ndarray, b: np.ndarray, label: str) -> tuple[np.ndarray, int, float]:
    # LAPACK least-squares avoids the instability from normal-equation inversion.
    beta, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank != a.shape[1] or singular.size == 0 or singular[-1] <= 0:
        raise NotIdentifiedError(f"{label}: rank-deficient design; remove collinear columns")
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(condition) or condition > 1e12:
        raise NotIdentifiedError(f"{label}: ill-conditioned design (condition={condition:.3g}); rescale or re-specify")
    return beta, int(rank), condition


def least_squares(
    x: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> LinearFit:
    """OLS/WLS using rank-checked LAPACK least-squares, no pseudoinverse."""
    design = _design(x, "OLS design")
    target = _outcome(y, design.shape[0])
    w = _weights(weights, design.shape[0])
    if w is None:
        beta, rank, condition = _solve(design, target, "OLS")
    else:
        sw = np.sqrt(w)
        beta, rank, condition = _solve(design * sw[:, None], target * sw, "WLS")
    fitted = design @ beta
    return LinearFit(beta, fitted, target - fitted, rank, condition, design.shape[0], design.shape[1])


def two_stage_least_squares(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[LinearFit, np.ndarray]:
    """2SLS/W2SLS fit for full structural X and full instrument matrix Z.

    Returns (fit with *structural* residuals, X_hat). When ``weights`` are supplied,
    X_hat is weighted-projected in the W^(1/2) metric so weighted 2SLS solves
    min_b (y - Xb)' W P_{W^{1/2} Z} W (y - Xb) without forming an N x N matrix.
    """
    structural = _design(x, "structural design X")
    instruments = _design(z, "instrument design Z")
    if structural.shape[0] != instruments.shape[0]:
        raise InvalidDesignError("Structural and instrument designs must have equal row counts")
    if instruments.shape[1] < structural.shape[1]:
        raise NotIdentifiedError("Fewer instrument columns than structural regressors")
    target = _outcome(y, structural.shape[0])
    w = _weights(weights, structural.shape[0])
    if w is None:
        gamma, _, _ = _solve(instruments, structural, "instrument matrix")
        projected = instruments @ gamma
        beta, rank, condition = _solve(projected, target, "projected structural design")
    else:
        sw = np.sqrt(w)
        gamma, _, _ = _solve(instruments * sw[:, None], structural * sw[:, None], "weighted instrument matrix")
        projected = instruments @ gamma
        beta, rank, condition = _solve(projected * sw[:, None], target * sw, "weighted projected structural design")
    fitted = structural @ beta
    return LinearFit(
        beta, fitted, target - fitted, rank, condition, structural.shape[0], structural.shape[1]
    ), projected
