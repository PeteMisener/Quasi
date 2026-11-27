"""Deterministic resampling & Wild Cluster Bootstrap utilities.

Includes IID/whole-cluster index resampling, empirical percentile intervals, and
Cameron-Gelbach-Miller (2008) / MacKinnon-Nielsen-Webb (2023) studentized wild
cluster bootstrap (Rademacher & Webb 6-point weights, restricted WCR and
unrestricted WCU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.inference.covariance import cluster_covariance
from quasi.inference.linear import least_squares

_WEBB_WEIGHTS = np.array(
    [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)],
    dtype=float,
)


def bootstrap_indices(
    n: int,
    *,
    reps: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Return reproducible row-index draws, with entire clusters kept together.

    Uses SeedSequence.spawn for independent repetition streams. Sampling G
    clusters with replacement may produce variable row counts for unequal sizes.
    """
    if (
        type(n) is not int
        or n < 2
        or type(reps) is not int
        or reps < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise InvalidDesignError("n>=2, reps>=1, seed>=0 must be integers")
    if groups is not None:
        labels = np.asarray(groups)
        if labels.ndim != 1 or len(labels) != n:
            raise InvalidDesignError("Bootstrap group labels must align with observations")
        import pandas as pd

        if pd.isna(labels).any():
            raise InvalidDesignError("Bootstrap group labels must not be missing")
        unique, inv = np.unique(labels, return_inverse=True)
        if len(unique) < 2:
            raise InsufficientSupportError("Cluster bootstrap needs at least two clusters")
        cluster_rows = [np.flatnonzero(inv == j) for j in range(len(unique))]
    children = np.random.SeedSequence(seed).spawn(reps)
    draws = []
    for child in children:
        rng = np.random.default_rng(child)
        if groups is None:
            draw = rng.integers(0, n, size=n, dtype=np.int64)
        else:
            sampled = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
            draw = np.concatenate([cluster_rows[j] for j in sampled])
        draws.append(draw)
    return tuple(draws)


def percentile_interval(draws: np.ndarray, *, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided empirical percentile CI; no automatic claim of validity."""
    a = np.asarray(draws, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise InvalidDesignError("Bootstrap estimates must contain >=2 finite draws")
    if not 0 < alpha < 1 or not np.isfinite(alpha):
        raise InvalidDesignError("alpha must be in (0,1)")
    q = np.quantile(a, [alpha / 2, 1 - alpha / 2])
    return float(q[0]), float(q[1])


@dataclass(frozen=True)
class WildBootstrapResult:
    """Studentized Wild Cluster Bootstrap hypothesis test output."""

    estimate: float
    null_value: float
    t_observed: float
    p_value: float
    n_clusters: int
    weight_type: str
    replications: int
    restricted: bool


def wild_cluster_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    param_index: int,
    *,
    null_value: float = 0.0,
    reps: int = 999,
    seed: int = 42,
    weights: Literal["auto", "rademacher", "webb"] = "auto",
    restricted: bool = True,
) -> WildBootstrapResult:
    """Studentized Wild Cluster Bootstrap for H0: beta[param_index] = null_value.

    Follows Cameron, Gelbach & Miller (2008) and MacKinnon, Nielsen & Webb (2023):
    - Defaults to Webb 6-point weights when G < 12 and Rademacher when G >= 12.
    - Imposes the null hypothesis (WCR, ``restricted=True``) by default for
      superior small-cluster size control.
    """
    if type(reps) is not int or reps < 49:
        raise InvalidDesignError("reps must be an integer >= 49")
    if type(seed) is not int or seed < 0:
        raise InvalidDesignError("seed must be a nonnegative integer")
    if weights not in ("auto", "rademacher", "webb"):
        raise InvalidDesignError("weights must be 'auto', 'rademacher', or 'webb'")
    if isinstance(null_value, bool) or not np.isscalar(null_value) or not np.isfinite(null_value):
        raise InvalidDesignError("null_value must be a finite scalar")

    fit = least_squares(x, y)
    n, p = fit.n_obs, fit.n_parameters
    if type(param_index) is not int or not (0 <= param_index < p):
        raise InvalidDesignError(f"param_index must be in [0, {p - 1}]")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) != n:
        raise InvalidDesignError("groups must match observation count")
    import pandas as pd

    if pd.isna(labels).any():
        raise InvalidDesignError("cluster labels cannot be missing")
    _, inv = np.unique(labels, return_inverse=True)
    g = int(inv.max() + 1)
    if g < 4:
        raise InsufficientSupportError("Wild cluster bootstrap requires at least 4 clusters")

    cov_obs = cluster_covariance(x_arr, fit.residuals, labels, small_sample=True)
    se_obs = float(np.sqrt(max(cov_obs[param_index, param_index], 0.0)))
    if se_obs <= 1e-14:
        raise InsufficientSupportError("Observed cluster SE is degenerate")
    est = float(fit.coefficients[param_index])
    t_obs = (est - float(null_value)) / se_obs

    # Build restricted or unrestricted baseline fit & residuals
    if restricted:
        if p == 1:
            y_hat0 = float(null_value) * x_arr[:, 0]
            res0 = y_arr - y_hat0
        else:
            keep = [j for j in range(p) if j != param_index]
            y_adj = y_arr - float(null_value) * x_arr[:, param_index]
            rfit = least_squares(x_arr[:, keep], y_adj)
            y_hat0 = rfit.fitted + float(null_value) * x_arr[:, param_index]
            res0 = y_arr - y_hat0
    else:
        y_hat0 = fit.fitted
        res0 = fit.residuals

    chosen_weights = ("webb" if g < 12 else "rademacher") if weights == "auto" else weights
    support = _WEBB_WEIGHTS if chosen_weights == "webb" else np.array([-1.0, 1.0], dtype=float)

    # Precompute QR factor for fast O(N p) OLS refits inside bootstrap loop
    q_mat, r_mat = np.linalg.qr(x_arr, mode="reduced")
    inv_r = np.linalg.solve(r_mat, np.eye(p))
    bread = inv_r @ inv_r.T
    scale_cr1 = (g / (g - 1.0)) * ((n - 1.0) / (n - p))
    target_center = float(null_value) if restricted else est

    rng = np.random.default_rng(seed)
    t_boot = np.empty(reps, dtype=float)
    for b in range(reps):
        w_cluster = rng.choice(support, size=g)
        y_star = y_hat0 + res0 * w_cluster[inv]
        beta_star = inv_r @ (q_mat.T @ y_star)
        u_star = y_star - x_arr @ beta_star
        scores = np.zeros((g, p), dtype=float)
        np.add.at(scores, inv, x_arr * u_star[:, None])
        cov_star = scale_cr1 * (bread @ (scores.T @ scores) @ bread)
        se_star = float(np.sqrt(max(cov_star[param_index, param_index], 1e-30)))
        t_boot[b] = (float(beta_star[param_index]) - target_center) / se_star

    p_val = float((1 + np.sum(np.abs(t_boot) >= abs(t_obs))) / (reps + 1))
    return WildBootstrapResult(
        estimate=est,
        null_value=float(null_value),
        t_observed=float(t_obs),
        p_value=p_val,
        n_clusters=g,
        weight_type=chosen_weights,
        replications=reps,
        restricted=restricted,
    )
