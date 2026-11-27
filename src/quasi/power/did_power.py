"""
Panel and Clustered Difference-in-Differences Power and MDE Calculator.

Implements exact within-unit variance calculations under AR(1) serial correlation
(Burlig, Preonas & Woerman 2020), compound symmetry, or user-supplied empirical
error covariance matrices, with small-cluster t-distribution degrees of freedom.
"""

from __future__ import annotations

from typing import Literal, Optional
import numpy as np
from scipy import stats

from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.power.base import PowerResult


def _panel_contrast_variance(
    n_pre: int,
    n_post: int,
    sd_y: float,
    serial_correlation: float,
    correlation_structure: Literal["ar1", "compound_symmetry"] = "ar1",
    error_cov: Optional[np.ndarray] = None,
) -> float:
    """Exact variance of (mean_post - mean_pre) for a single unit/cluster."""
    total_t = n_pre + n_post
    contrast = np.concatenate([
        np.full(n_pre, -1.0 / n_pre, dtype=float),
        np.full(n_post, 1.0 / n_post, dtype=float),
    ])
    if error_cov is not None:
        cov = np.asarray(error_cov, dtype=float)
        if cov.shape != (total_t, total_t) or not np.isfinite(cov).all():
            raise InvalidDesignError(f"error_cov must be a finite ({total_t}, {total_t}) matrix")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise InvalidDesignError("error_cov must be symmetric")
        val = float(contrast @ cov @ contrast)
        if val <= 0:
            raise InvalidDesignError("error_cov produces nonpositive DiD contrast variance")
        return val

    if correlation_structure == "compound_symmetry":
        # Sigma_ij = sigma^2 * (rho + (1 - rho) * I(i == j)) => c' Sigma c = sigma^2 * (1 - rho) * (1/T0 + 1/T1)
        return float((sd_y**2) * (1.0 - serial_correlation) * (1.0 / n_pre + 1.0 / n_post))
    if correlation_structure == "ar1":
        # Exact Burlig-Preonas-Woerman (2020) AR(1) covariance: Sigma_ij = sigma^2 * rho^|i - j|
        idx = np.arange(total_t)
        cov = (sd_y**2) * (serial_correlation ** np.abs(idx[:, None] - idx[None, :]))
        return float(contrast @ cov @ contrast)
    raise InvalidDesignError("correlation_structure must be 'ar1' or 'compound_symmetry'")


def did_mde(
    n_clusters: int,
    n_periods: int,
    pre_periods: int,
    cluster_size: int = 1,
    treated_fraction: float = 0.5,
    icc: float = 0.0,
    serial_correlation: float = 0.5,
    sd_y: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    *,
    correlation_structure: Literal["ar1", "compound_symmetry"] = "ar1",
    error_cov: Optional[np.ndarray] = None,
) -> PowerResult:
    """
    Computes Minimum Detectable Effect (MDE) for panel Difference-in-Differences.

    Uses exact contrast variance c' Sigma c under AR(1) serial correlation
    (default; Burlig, Preonas & Woerman 2020), compound symmetry, or a supplied
    empirical T x T residual covariance matrix ``error_cov``.
    """
    G = n_clusters
    T = n_periods
    T0 = pre_periods
    T1 = T - T0
    m = cluster_size
    P = treated_fraction
    rho = serial_correlation
    sigma = sd_y

    if T0 <= 0 or T1 <= 0:
        raise InvalidDesignError("Must have at least 1 pre-period and 1 post-period.")
    if G < 4:
        raise InvalidDesignError("Must have at least 4 clusters for clustered panel inference.")
    if m < 1:
        raise InvalidDesignError("cluster_size must be at least 1.")
    if not (0 < P < 1):
        raise InvalidDesignError("treated_fraction must be strictly between 0 and 1.")
    if not (-1.0 < rho < 1.0):
        raise InvalidDesignError("serial_correlation must lie strictly between -1 and 1.")
    if not (0.0 <= icc <= 1.0):
        raise InvalidDesignError("icc must lie between 0 and 1.")
    if sigma <= 0 or not np.isfinite(sigma):
        raise InvalidDesignError("sd_y must be positive and finite.")
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise InvalidDesignError("alpha and power must lie strictly between 0 and 1.")

    # Small-sample degrees of freedom: G - 2
    df = max(1, G - 2)
    t_alpha = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    t_power = float(stats.t.ppf(power, df=df))
    M = t_alpha + t_power

    # Design Effect for clustering: VIF = 1 + (m - 1) * icc
    vif_cluster = 1.0 + (m - 1) * icc

    contrast_var = _panel_contrast_variance(
        T0, T1, sigma, rho, correlation_structure=correlation_structure, error_cov=error_cov
    )
    var_did = (contrast_var * vif_cluster) / (P * (1.0 - P) * G * m)
    se_did = float(np.sqrt(max(var_did, 0.0)))
    if se_did <= 0.0:
        raise InvalidDesignError("Implied DiD variance is nonpositive")

    mde_val = float(M * se_did)
    mde_std = float(mde_val / sigma)

    def curve_effect(effects: np.ndarray) -> np.ndarray:
        ncp = np.abs(effects) / se_did
        return 1.0 - stats.nct.cdf(t_alpha, df=df, nc=ncp) + stats.nct.cdf(-t_alpha, df=df, nc=ncp)

    def curve_sample(clusters: np.ndarray) -> np.ndarray:
        out = []
        for g_val in clusters:
            df_g = max(1, int(g_val) - 2)
            t_a = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_g))
            v_g = (contrast_var * vif_cluster) / (P * (1.0 - P) * g_val * m)
            se_g = float(np.sqrt(max(v_g, 0.0)))
            ncp_g = mde_val / se_g
            p = 1.0 - stats.nct.cdf(t_a, df=df_g, nc=ncp_g) + stats.nct.cdf(-t_a, df=df_g, nc=ncp_g)
            out.append(p)
        return np.array(out)

    params = {
        "Cluster Count (G)": G,
        "Total Periods (T)": T,
        "Pre-Periods (T0)": T0,
        "Post-Periods (T1)": T1,
        "Cluster Size (m)": m,
        "Treated Share (P)": P,
        "Correlation Structure": "empirical_cov" if error_cov is not None else correlation_structure,
        "Serial Correlation (ρ)": rho,
        "Intracluster Corr (ICC)": icc,
        "Within-Unit Contrast Var": contrast_var,
        "Degrees of Freedom (G - 2)": df,
        "Standard Error (SE)": se_did,
    }

    return PowerResult(
        design="Difference-in-Differences (Clustered Panel)",
        mde=mde_val,
        mde_standardized=mde_std,
        target_power=power,
        alpha=alpha,
        parameters=params,
        curve_func_effect=curve_effect,
        curve_func_sample=curve_sample,
        sample_param_name="Cluster Count (G)",
        sample_param_val=float(G),
    )


def did_sample_size(
    target_mde: float,
    n_periods: int,
    pre_periods: int,
    cluster_size: int = 1,
    treated_fraction: float = 0.5,
    icc: float = 0.0,
    serial_correlation: float = 0.5,
    sd_y: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    max_clusters: int = 10000,
    *,
    correlation_structure: Literal["ar1", "compound_symmetry"] = "ar1",
    error_cov: Optional[np.ndarray] = None,
) -> int:
    """Solves for minimum required cluster count G to detect target_mde.

    Fails closed with InsufficientSupportError if target_mde cannot be reached
    within ``max_clusters``.
    """
    if not np.isfinite(target_mde) or abs(target_mde) <= 0:
        raise InvalidDesignError("target_mde must be a finite nonzero number.")
    if max_clusters < 4:
        raise InvalidDesignError("max_clusters must be at least 4.")

    low, high = 4, max_clusters
    best: Optional[int] = None
    while low <= high:
        mid = (low + high) // 2
        res = did_mde(
            n_clusters=mid,
            n_periods=n_periods,
            pre_periods=pre_periods,
            cluster_size=cluster_size,
            treated_fraction=treated_fraction,
            icc=icc,
            serial_correlation=serial_correlation,
            sd_y=sd_y,
            alpha=alpha,
            power=power,
            correlation_structure=correlation_structure,
            error_cov=error_cov,
        )
        if res.mde <= abs(target_mde):
            best = mid
            high = mid - 1
        else:
            low = mid + 1
    if best is None:
        raise InsufficientSupportError(
            f"Target MDE={abs(target_mde):.4g} is infeasible within max_clusters={max_clusters}; "
            "increase max_clusters, cluster_size, or periods."
        )
    return best
