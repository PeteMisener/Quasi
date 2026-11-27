"""
Regression Discontinuity Design (RDD) Power and MDE Calculator.

Uses exact asymptotic boundary local-polynomial variance constants
(Calonico, Cattaneo & Titiunik 2014; Cattaneo, Titiunik & Vazquez-Bare 2019)
for triangular, Epanechnikov, and uniform kernels under conventional or
robust bias-corrected (RBC) inference.
"""

from __future__ import annotations

from typing import Literal, Optional
import numpy as np
from scipy import stats

from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.power.base import PowerResult

# Exact asymptotic boundary variance factors V_{p,K} = e_0' Gamma_p^{-1} Psi_p Gamma_p^{-1} e_0
# for one-sided local polynomial regression of order p on [0, 1] with kernel K:
# - Uniform (order 1): 4.0, (order 2): 9.0
# - Triangular (order 1): 4.8, (order 2): 10.2857
# - Epanechnikov (order 1): 4.4926, (order 2): 9.75
# Normalized relative to a sharp balanced local mean (factor 4 = 2 per side => ratio = V / 4):
_BOUNDARY_VAR_FACTOR = {
    (1, "uniform"): 4.0,
    (1, "triangular"): 4.8,
    (1, "epanechnikov"): 4.497982,
    (2, "uniform"): 9.0,
    (2, "triangular"): 10.285714,
    (2, "epanechnikov"): 9.816468,
}
# Approximate RBC variance inflation factor when rho = h / b = 1 (CCT 2014 / rdpower)
_RBC_INFLATION = {1: 1.50, 2: 1.45}


def rdd_mde(
    n_total: int,
    fraction_in_bandwidth: float = 0.25,
    order: int = 1,
    kernel: str = "triangular",
    sd_y: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    *,
    inference: Literal["conventional", "bias_corrected"] = "conventional",
) -> PowerResult:
    """
    Computes Minimum Detectable Effect (MDE) for local polynomial Regression Discontinuity.
    """
    if n_total < 30:
        raise InvalidDesignError("Sample size too small for meaningful RDD power analysis (need n_total >= 30).")
    if not (0.01 <= fraction_in_bandwidth <= 1.0):
        raise InvalidDesignError("fraction_in_bandwidth must be between 0.01 and 1.0.")
    if order not in (1, 2):
        raise InvalidDesignError("order must be 1 (local linear) or 2 (local quadratic).")
    kern = kernel.lower()
    if (order, kern) not in _BOUNDARY_VAR_FACTOR:
        raise InvalidDesignError("kernel must be 'triangular', 'epanechnikov', or 'uniform'.")
    if inference not in ("conventional", "bias_corrected"):
        raise InvalidDesignError("inference must be 'conventional' or 'bias_corrected'.")
    if sd_y <= 0 or not np.isfinite(sd_y):
        raise InvalidDesignError("sd_y must be positive and finite.")
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise InvalidDesignError("alpha and power must lie strictly between 0 and 1.")

    n_eff = int(n_total * fraction_in_bandwidth)
    if n_eff < 10:
        raise InvalidDesignError("Effective sample within bandwidth is fewer than 10 observations.")

    # Each side has ~n_eff / 2 observations; Var(hat_mu_+ - hat_mu_-) = (V_{p,K} * sigma^2 / (n_eff/2)) * 2
    # = 4 * V_{p,K} * sigma^2 / n_eff (for uniform p=0 mean difference: V=1 => 4*sigma^2/n_eff).
    v_pk = _BOUNDARY_VAR_FACTOR[(order, kern)]
    rbc_mult = _RBC_INFLATION[order] if inference == "bias_corrected" else 1.0
    c_p = v_pk * rbc_mult
    var_rdd = (4.0 * (sd_y**2) * c_p) / n_eff
    se_rdd = float(np.sqrt(var_rdd))

    df = max(10, n_eff - 2 * (order + 1))
    t_alpha = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    t_power = float(stats.t.ppf(power, df=df))
    M = t_alpha + t_power

    mde_val = float(M * se_rdd)
    mde_std = float(mde_val / sd_y)

    def curve_effect(effects: np.ndarray) -> np.ndarray:
        ncp = np.abs(effects) / se_rdd
        return 1.0 - stats.t.cdf(t_alpha, df=df, loc=ncp) + stats.t.cdf(-t_alpha, df=df, loc=ncp)

    def curve_sample(samples: np.ndarray) -> np.ndarray:
        out = []
        for n_val in samples:
            ne = max(10, int(n_val * fraction_in_bandwidth))
            df_n = max(10, ne - 2 * (order + 1))
            ta = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_n))
            se_n = float(np.sqrt((4.0 * (sd_y**2) * c_p) / ne))
            ncp_n = mde_val / se_n
            p = 1.0 - stats.t.cdf(ta, df=df_n, loc=ncp_n) + stats.t.cdf(-ta, df=df_n, loc=ncp_n)
            out.append(p)
        return np.array(out)

    params = {
        "Total Sample (N)": n_total,
        "Bandwidth Fraction": fraction_in_bandwidth,
        "Effective N in Bandwidth": n_eff,
        "Polynomial Degree": order,
        "Kernel Weighting": kern.capitalize(),
        "Inference": inference,
        "Boundary Variance Factor (V_p,K * RBC)": c_p,
        "Standard Error (SE)": se_rdd,
    }

    return PowerResult(
        design=f"Regression Discontinuity (Order {order}, {kern.capitalize()}, {inference})",
        mde=mde_val,
        mde_standardized=mde_std,
        target_power=power,
        alpha=alpha,
        parameters=params,
        curve_func_effect=curve_effect,
        curve_func_sample=curve_sample,
        sample_param_name="Total Sample Size (N)",
        sample_param_val=float(n_total),
    )


def rdd_sample_size(
    target_mde: float,
    fraction_in_bandwidth: float = 0.25,
    order: int = 1,
    kernel: str = "triangular",
    sd_y: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    max_n: int = 500000,
    *,
    inference: Literal["conventional", "bias_corrected"] = "conventional",
) -> int:
    """Solves for total required sample size N to detect target_mde at the cutoff.

    Raises InsufficientSupportError if target_mde is not achievable within max_n.
    """
    if not np.isfinite(target_mde) or abs(target_mde) <= 0:
        raise InvalidDesignError("target_mde must be a finite nonzero number.")
    min_n = max(30, int(np.ceil(10.0 / fraction_in_bandwidth)))
    if max_n < min_n:
        raise InvalidDesignError(f"max_n must be at least {min_n}.")

    low = min_n
    high = max_n
    best: Optional[int] = None
    while low <= high:
        mid = (low + high) // 2
        res = rdd_mde(
            n_total=mid,
            fraction_in_bandwidth=fraction_in_bandwidth,
            order=order,
            kernel=kernel,
            sd_y=sd_y,
            alpha=alpha,
            power=power,
            inference=inference,
        )
        if res.mde <= abs(target_mde):
            best = mid
            high = mid - 1
        else:
            low = mid + 1
    if best is None:
        raise InsufficientSupportError(
            f"Target MDE={abs(target_mde):.4g} is infeasible within max_n={max_n}; increase max_n."
        )
    return best
