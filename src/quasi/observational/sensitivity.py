"""Cinelli & Hazlett (2020, JRSS-B) Omitted-Variable Bias (OVB) Sensitivity Analysis.

Quantifies how strong an unobserved confounder Z (orthogonal to observed controls X)
would need to be—measured in partial R^2 terms (R^2_{D~Z|X}, R^2_{Y~Z|D,X})—to
reduce an estimated treatment effect by fraction q or render it statistically
insignificant at level alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats

from quasi.errors import InvalidDesignError


@dataclass(frozen=True)
class OVBSensitivityResult:
    """Cinelli & Hazlett (2020) robustness values and adjusted-effect evaluator."""

    estimate: float
    se: float
    t_statistic: float
    df: int
    partial_r2_yd_x: float
    rv_q: float
    rv_q_alpha: float
    q: float
    alpha: float

    def adjusted_estimate(
        self,
        r2dz_x: float,
        r2yz_dx: float,
        *,
        direction: int = -1,
    ) -> tuple[float, float, tuple[float, float]]:
        """Compute bias-adjusted (estimate, se, ci) for hypothesized confounder strengths.

        Parameters
        ----------
        r2dz_x : partial R^2 of confounder Z with treatment D given X, in [0, 1).
        r2yz_dx : partial R^2 of confounder Z with outcome Y given (D, X), in [0, 1).
        direction : -1 (bias pulls estimate toward zero, worst case) or +1.
        """
        if not (0.0 <= r2dz_x < 1.0) or not (0.0 <= r2yz_dx < 1.0):
            raise InvalidDesignError("Partial R^2 values must lie in [0, 1)")
        if direction not in (-1, 1):
            raise InvalidDesignError("direction must be -1 or +1")
        # Cinelli & Hazlett (2020) Theorem 2:
        # |bias| = se * sqrt(df) * sqrt((r2yz_dx * r2dz_x) / (1 - r2dz_x))
        # se_adj = se * sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * sqrt(df / (df - 1))
        bias_mag = self.se * np.sqrt(self.df) * np.sqrt((r2yz_dx * r2dz_x) / (1.0 - r2dz_x))
        sign = np.sign(self.estimate) if self.estimate != 0 else 1.0
        est_adj = float(self.estimate + direction * sign * bias_mag)
        se_adj = float(self.se * np.sqrt(((1.0 - r2yz_dx) / (1.0 - r2dz_x)) * (self.df / max(1, self.df - 1))))
        crit = float(stats.t.ppf(1.0 - self.alpha / 2.0, df=max(1, self.df - 1)))
        return est_adj, se_adj, (est_adj - crit * se_adj, est_adj + crit * se_adj)


def _robustness_value(f_val: float) -> float:
    """Cinelli & Hazlett (2020) eq. (16): RV = 0.5 * (sqrt(f^4 + 4 f^2) - f^2)."""
    if f_val <= 0.0:
        return 0.0
    f2 = f_val * f_val
    return float(0.5 * (np.sqrt(f2 * f2 + 4.0 * f2) - f2))


def ovb_sensitivity(
    estimate: float,
    se: float,
    df: int,
    *,
    q: float = 1.0,
    alpha: float = 0.05,
) -> OVBSensitivityResult:
    """Compute Cinelli & Hazlett (2020) partial R^2 and Robustness Values RV_q, RV_{q,alpha}."""
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
        raise InvalidDesignError("estimate must be finite and se strictly positive")
    if type(df) is not int or df < 2:
        raise InvalidDesignError("df must be an integer >= 2")
    if not (0.0 < q <= 1.0) or not (0.0 < alpha < 1.0):
        raise InvalidDesignError("q must be in (0, 1] and alpha in (0, 1)")

    t_stat = float(abs(estimate / se))
    r2_yd_x = float((t_stat * t_stat) / (t_stat * t_stat + df))
    f_yd_x = float(t_stat / np.sqrt(df))

    # RV_q: minimum equal partial R^2 to reduce estimate by fraction q
    fq = q * f_yd_x
    rv_q = _robustness_value(fq)

    # RV_{q, alpha}: minimum equal partial R^2 to render reduced effect insignificant at alpha
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df - 1))
    f_crit = t_crit / np.sqrt(df - 1)
    fq_alpha = max(0.0, fq - f_crit)
    rv_q_alpha = _robustness_value(fq_alpha)

    return OVBSensitivityResult(
        estimate=float(estimate),
        se=float(se),
        t_statistic=float(estimate / se),
        df=df,
        partial_r2_yd_x=r2_yd_x,
        rv_q=rv_q,
        rv_q_alpha=rv_q_alpha,
        q=float(q),
        alpha=float(alpha),
    )
