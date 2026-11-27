"""
Two-Stage Least Squares (2SLS) Instrumental Variables Estimator.
Provides conventional and robust first-stage diagnostics, explicit robust or
clustered covariance, and experimental Anderson-Rubin null tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.prep.extractor import extract_column_arrays
from quasi.inference import least_squares, two_stage_least_squares
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.iv.inference import (
    anderson_rubin_confidence_set,
    anderson_rubin_test,
    covariance_for_iv,
    effective_f_statistic,
    excluded_instruments_test,
    gmm_hansen_j,
    liml_estimate,
    moreira_clr_test,
    tf_critical_value,
)


class IVResult(QEDResult):
    """Result container for Two-Stage Least Squares models."""

    def __init__(
        self,
        estimate: float,
        se: float,
        p_value: float,
        ci: Tuple[float, float],
        outcome_name: str,
        endogenous_name: str,
        instruments: List[str],
        first_stage_f: float,
        sargan_j_p: Optional[float],
        n_obs: int,
        diagnostics: Dict[str, DiagnosticTest],
        underlying_model: Optional[BaseQEDModel] = None,
        *, covariance: str = "HC0", n_clusters: int | None = None,
        first_stage_robust: Any = None, first_stage_partial_r2: float = 0.0,
        ar_null_zero: Any = None,
    ) -> None:
        super().__init__(
            estimate=estimate,
            se=se,
            p_value=p_value,
            ci=ci,
            estimand="IV structural coefficient (causal interpretation requires valid instrument assumptions)",
            outcome_name=outcome_name,
            model_name="TwoStageLS",
            n_obs=n_obs,
            n_treated=n_obs,
            diagnostics=diagnostics,
            sub_effects={
                "endogenous_variable": endogenous_name,
                "instruments": instruments,
                "first_stage_f_statistic": first_stage_f,
                "sargan_j_p_value": sargan_j_p,
                "first_stage_robust_statistic": first_stage_robust.statistic,
                "first_stage_robust_p_value": first_stage_robust.p_value,
                "first_stage_robust_distribution": first_stage_robust.distribution,
                "first_stage_partial_r_squared": first_stage_partial_r2,
                "anderson_rubin_null_zero": {"statistic": ar_null_zero.statistic,
                    "p_value": ar_null_zero.p_value, "distribution": ar_null_zero.distribution},
            },
            model_meta={"instruments": instruments},
            underlying_model=underlying_model,
        )
        self.endogenous_name = endogenous_name
        self.instruments = instruments
        self.first_stage_f = first_stage_f
        self.sargan_j_p = sargan_j_p
        self.inference_details = {"method": f"IV_2SLS_{covariance}", "covariance": covariance,
                                  "small_sample": covariance in ("HC1", "CR1"), "reference_certified": False,
                                  "reference_distribution": "t" if n_clusters is not None else "normal",
                                  "clusters": n_clusters,
                                  "weak_instrument_robust_ci": False,
                                  "ar_set_available_for_single_instrument": len(instruments) == 1}
        self.covariance = covariance
        self.n_clusters = n_clusters
        self.first_stage_robust = first_stage_robust
        self.first_stage_partial_r2 = first_stage_partial_r2
        self.ar_null_zero = ar_null_zero

    def ar_test(self, null_effect: float):
        """Experimental robust Anderson-Rubin test for H0: endogenous coefficient = value.

        Returns WaldTest; a single rejection is not a weak-IV-robust CI.
        """
        if self._model is None:
            raise InvalidDesignError("AR test requires the fitted model and original data")
        m = self._model
        y, x = m.arrays["outcome"], m.arrays["endogenous"]
        z = m.arrays["instruments"]
        z = z.reshape(-1, 1) if z.ndim == 1 else z
        blocks = [np.ones(len(y)), z]
        if m.covariates:
            w = m.arrays["covariates"]
            blocks.append(w.reshape(-1, 1) if w.ndim == 1 else w)
        return anderson_rubin_test(y, x, np.column_stack(blocks), z.shape[1],
                                  null_effect, kind=m.covariance, groups=m.arrays.get("cluster"))

    def ar_confidence_set(self, *, alpha: float | None = None):
        """Experimental weak-IV-robust AR confidence set (exactly one instrument).

        The returned set can be unbounded, disconnected, or empty and MUST NOT
        be confused with the standard finite Wald `ci` attribute. The test
        assumes valid instruments and uses asymptotic HC/cluster calibration.
        """
        if self._model is None:
            raise InvalidDesignError("AR inversion requires fitted model and original data")
        m = self._model
        y, x, z = m.arrays["outcome"], m.arrays["endogenous"], m.arrays["instruments"]
        z = z.reshape(-1, 1) if z.ndim == 1 else z
        if z.shape[1] != 1:
            raise InvalidDesignError("Analytic AR inversion currently supports one excluded instrument")
        blocks = [np.ones(len(y)), z]
        if m.covariates:
            w = m.arrays["covariates"]
            blocks.append(w.reshape(-1, 1) if w.ndim == 1 else w)
        return anderson_rubin_confidence_set(y, x, np.column_stack(blocks), 1,
            alpha=m.alpha if alpha is None else alpha, kind=m.covariance,
            groups=m.arrays.get("cluster"))

    def _exog_blocks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._model is None:
            raise InvalidDesignError("Requires fitted model and original data")
        m = self._model
        y, x_end, z_inst = m.arrays["outcome"], m.arrays["endogenous"], m.arrays["instruments"]
        z_inst = z_inst.reshape(-1, 1) if z_inst.ndim == 1 else z_inst
        w_blocks = [np.ones((len(y), 1), dtype=float)]
        if m.covariates:
            w = m.arrays["covariates"]
            w_blocks.append(w.reshape(-1, 1) if w.ndim == 1 else w)
        return y, x_end, z_inst, np.column_stack(w_blocks)

    def effective_f(self):
        """Montiel Olea & Pflueger (2013) robust/clustered Effective F-statistic."""
        _, x_end, z_inst, w_ctrl = self._exog_blocks()
        m = self._model
        return effective_f_statistic(x_end, z_inst, w_ctrl, kind=m.covariance, groups=m.arrays.get("cluster"))

    def liml(self, *, fuller_alpha: float = 0.0):
        """LIML (fuller_alpha=0) or Fuller(fuller_alpha) k-class IV estimate."""
        y, x_end, z_inst, w_ctrl = self._exog_blocks()
        m = self._model
        return liml_estimate(
            y, x_end, z_inst, w_ctrl,
            fuller_alpha=fuller_alpha, alpha=m.alpha, kind=m.covariance, groups=m.arrays.get("cluster"),
        )

    def clr_test(self, null_effect: float = 0.0, *, reps: int = 4000, seed: int = 42):
        """Moreira (2003) Conditional Likelihood Ratio weak-IV-robust test (homoskedastic/unclustered)."""
        if self._model is not None and self._model.cluster is not None:
            from quasi.errors import UnsupportedInferenceError
            raise UnsupportedInferenceError(
                "Moreira CLR test assumes homoskedastic unclustered errors; use anderson_rubin_test() for clustered IV"
            )
        y, x_end, z_inst, w_ctrl = self._exog_blocks()
        return moreira_clr_test(y, x_end, z_inst, w_ctrl, null_effect=null_effect, reps=reps, seed=seed)

    def tf_confidence_interval(self, *, alpha: float | None = None) -> tuple[float, float]:
        """Lee, McCrary, Moreira & Porter (2022, AER) tF CI for just-identified IV."""
        if len(self.instruments) != 1:
            raise InvalidDesignError("tF adjustment applies to just-identified (single-instrument) IV")
        eff = self.effective_f()
        lvl = (1.0 - self.confidence_level) if alpha is None else alpha
        crit = tf_critical_value(eff.f_effective, alpha=lvl)
        if not np.isfinite(crit):
            return (-float("inf"), float("inf"))
        return (self.estimate - crit * self.se, self.estimate + crit * self.se)

    def gmm_overidentification_test(self, *, allow_clustered: bool = False):
        """Experimental auxiliary two-step robust (HC0/HC1 or opt-in clustered CR0) GMM Hansen J."""
        if self._model is None:
            raise InvalidDesignError('GMM J requires fitted model and original data')
        m = self._model
        if m.cluster is not None and not allow_clustered:
            raise InvalidDesignError('Clustered GMM overidentification inference is not implemented')
        y, x_end, z_inst = m.arrays['outcome'], m.arrays['endogenous'], m.arrays['instruments']
        z_inst = z_inst.reshape(-1, 1) if z_inst.ndim == 1 else z_inst
        n = len(y)
        xb = [np.ones(n), x_end.reshape(-1, 1)]
        zb = [np.ones(n), z_inst]
        if m.covariates:
            w = m.arrays['covariates']
            w = w.reshape(-1, 1) if w.ndim == 1 else w
            xb.append(w)
            zb.append(w)
        return gmm_hansen_j(
            y, np.column_stack(xb), np.column_stack(zb),
            groups=m.arrays.get('cluster') if allow_clustered else None,
        )

    def summary(self) -> str:
        inst_str = ", ".join(self.instruments)
        lines = [
            "[EXPERIMENTAL] Research prototype; reference validation pending.",
            "=" * 70,
            f"Two-Stage Least Squares (2SLS) | Outcome: {self.outcome_name}",
            "=" * 70,
            f"Endogenous Variable: {self.endogenous_name}",
            f"Instruments:         {inst_str}",
            f"2SLS coefficient:    {self.estimate:+.4f} ({self.covariance} SE: {self.se:.4f})",
            f"p-value:             {self.p_value:.4f}",
            f"{100*self.confidence_level:.0f}% Conf. Interval:  [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"First-Stage F-Stat:  {self.first_stage_f:.2f} (conventional; not strength certification)",
            f"First-Stage Robust:  {self.first_stage_robust.statistic:.2f} ({self.first_stage_robust.distribution} test)",
            f"AR null beta=0:      p = {self.ar_null_zero.p_value:.4f} (weak-IV-robust test)",
            f"Observations:        {self.n_obs:,}",
            "-" * 70,
            "Instrument Validity Diagnostics:",
        ]
        for t in self.diagnostics.values():
            p_str = f"p = {t.p_value:.4f}" if t.p_value is not None else "-"
            lines.append(f"  • {t.name:<30} [{t.status.value}] {t.statistic_name} = {t.statistic_value:.3f} ({p_str})")
        lines.append("=" * 70)
        return "\n".join(lines)

    def plot(self, **kwargs: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=100)
        ax.errorbar(
            [0],
            [self.estimate],
            yerr=[[self.estimate - self.ci_lower], [self.ci_upper - self.estimate]],
            fmt="o",
            color="#1B4F72",
            ecolor="#1B4F72",
            elinewidth=2.0,
            capsize=4.0,
            label=f"2SLS coefficient ({self.endogenous_name})",
        )
        ax.axhline(0, color="#7F8C8D", linestyle="--", linewidth=0.8)
        ax.set_xticks([0])
        ax.set_xticklabels([self.endogenous_name])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")
        ax.set_ylabel("IV coefficient", fontsize=10, fontweight="medium")
        ax.set_title(f"2SLS Estimate for {self.outcome_name}", fontsize=11, fontweight="bold", pad=10)
        fig.tight_layout()
        return fig, ax

    def bibtex(self) -> str:
        return """@article{angrist1996identification,
  title={Identification of causal effects using instrumental variables},
  author={Angrist, Joshua D and Imbens, Guido W and Rubin, Donald B},
  journal={Journal of the American Statistical Association},
  volume={91},
  number={434},
  pages={444--455},
  year={1996}
}"""

    def methods_text(self) -> str:
        return (
            f"An experimental 2SLS coefficient for '{self.endogenous_name}' on '{self.outcome_name}' was estimated via "
            f"Two-Stage Least Squares (2SLS) using instruments: {', '.join(self.instruments)}. "
            f"First-stage relevance was screened using a conventional partial F-statistic (F = {self.first_stage_f:.2f}). "
            f"The {self.covariance} sandwich covariance produces a coefficient of "
            f"{self.estimate:.4f} (p = {self.p_value:.4f})."
        )


class TwoStageLS(BaseQEDModel):
    """
    Two-Stage Least Squares (2SLS) Estimator.

    Parameters:
        data: DataFrame.
        outcome: Dependent variable Y.
        endogenous: Endogenous regressor X.
        instruments: Exogenous excluded instruments Z.
        covariates: Exogenous included controls W.
        cluster: Optional cluster variable for cluster-robust standard errors.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, Any],
        outcome: str,
        endogenous: str,
        instruments: List[str],
        covariates: Optional[List[str]] = None,
        cluster: Optional[str] = None,
        alpha: float = 0.05,
        covariance: Optional[str] = None,
    ) -> None:
        covariance = ("CR1" if cluster is not None else "HC0") if covariance is None else covariance
        if covariance not in (("CR0", "CR1") if cluster is not None else ("HC0", "HC1")):
            raise InvalidDesignError("IV covariance must be HC0/HC1 without cluster or CR0/CR1 with cluster")
        self.covariance = covariance
        super().__init__(data=data, outcome=outcome, covariates=covariates, cluster=cluster, alpha=alpha)
        self.endogenous = endogenous
        if not isinstance(instruments, (list, tuple)) or not instruments:
            raise InvalidDesignError("At least one excluded IV instrument is required")
        self.instruments = list(instruments)

        cols: Dict[str, Union[str, List[str], None]] = {
            "outcome": self.outcome,
            "endogenous": self.endogenous,
            "instruments": self.instruments,
            "covariates": self.covariates if self.covariates else None,
            "cluster": self.cluster,
        }
        self.arrays = extract_column_arrays(data, cols)

    def fit(self) -> IVResult:
        Y = self.arrays["outcome"]
        X_end = self.arrays["endogenous"]
        Z_inst = self.arrays["instruments"]
        N = len(Y)
        groups = self.arrays.get("cluster")
        g_count = None if groups is None else len(np.unique(groups))
        if g_count is not None and g_count < 10:
            raise InsufficientSupportError("Clustered IV requires at least 10 distinct clusters; few-cluster inference is not supported")

        if len(Z_inst.shape) == 1:
            Z_inst = Z_inst.reshape(-1, 1)

        # Design matrix Z_full: [1, Z, W] (all exogenous variables)
        z_cols = [np.ones(N), Z_inst]
        # Design matrix X_full: [1, X_end, W] (structural regressors)
        x_cols = [np.ones(N), X_end.reshape(-1, 1)]

        if self.covariates and "covariates" in self.arrays:
            W = self.arrays["covariates"]
            if len(W.shape) == 1:
                W = W.reshape(-1, 1)
            z_cols.append(W)
            x_cols.append(W)

        Z_full = np.column_stack(z_cols)
        X_full = np.column_stack(x_cols)

        # Rank-safe 2SLS without an NxN projection matrix. First-stage F is
        # conventional and not a weak-IV robust diagnostic under heteroskedasticity.
        first_stage = least_squares(Z_full, X_end)
        z_rest_cols = [np.ones(N)]
        if self.covariates and "covariates" in self.arrays:
            z_rest_cols.append(W)
        Z_rest = np.column_stack(z_rest_cols)
        restricted = least_squares(Z_rest, X_end)
        k_inst = Z_inst.shape[1]
        df_resid = N - Z_full.shape[1]
        if df_resid <= 0:
            raise InsufficientSupportError("IV first stage has no residual degrees of freedom")
        ssr_unrestricted = float(np.sum(first_stage.residuals**2))
        ssr_restricted = float(np.sum(restricted.residuals**2))
        if ssr_unrestricted <= np.finfo(float).eps * max(1.0, float(X_end @ X_end)):
            raise InsufficientSupportError("IV first-stage residual variance is effectively zero")
        f_stat = float(max(0.0, (ssr_restricted - ssr_unrestricted) / k_inst)
                       / (ssr_unrestricted / df_resid))
        partial_r2 = float(max(0., min(1., (ssr_restricted - ssr_unrestricted) / ssr_restricted)))
        first_stage_cov = covariance_for_iv(Z_full, first_stage.residuals,
                                            kind=self.covariance, groups=groups)
        robust_first = excluded_instruments_test(first_stage.coefficients, first_stage_cov,
                                                  k_inst, groups)

        fit, X_projected = two_stage_least_squares(X_full, Z_full, Y)
        tau_estimate = float(fit.coefficients[1])
        residuals = fit.residuals  # structural Y - X beta, never fitted first-stage X
        # For linear 2SLS: (Xhat'X)^-1 [Xhat' diag(u^2) Xhat] (X'Xhat)^-1.
        # Since Xhat=Pz X, Xhat'X = Xhat'Xhat: shared HC0 regression scores apply.
        cov_matrix = covariance_for_iv(X_projected, residuals,
                                       kind=self.covariance, groups=groups)
        variance = float(cov_matrix[1, 1])
        if variance <= 0 or not np.isfinite(variance):
            raise InsufficientSupportError("IV HC0 variance is nonpositive; inference unavailable")
        se_estimate = float(np.sqrt(variance))

        z_val = tau_estimate / se_estimate if se_estimate > 0 else 0.0
        p_val = float(2.0 * (stats.t.sf(abs(z_val), g_count - 1) if g_count is not None
                               else stats.norm.sf(abs(z_val))))
        crit = float((stats.t.ppf(1.0-self.alpha/2.0, g_count-1) if g_count is not None
                      else stats.norm.ppf(1.0-self.alpha/2.0)))
        ar_zero = anderson_rubin_test(Y, X_end, Z_full, k_inst, 0.0,
                                     kind=self.covariance, groups=groups)
        ci = (tau_estimate - crit * se_estimate, tau_estimate + crit * se_estimate)

        # Diagnostics: Weak instruments + Sargan overidentification test
        diagnostics: Dict[str, DiagnosticTest] = {}

        # Statistical significance of the first stage is not a strength certificate.
        diagnostics["First-stage conditional relevance"] = DiagnosticTest(
            name="Excluded-instrument relevance (robust Wald)",
            status=DiagnosticStatus.WARN,
            statistic_name=f"{robust_first.distribution}({robust_first.df_numerator})",
            statistic_value=robust_first.statistic,
            p_value=robust_first.p_value,
            threshold="No universal robust weak-IV threshold",
            message="The robust first-stage test is not a weak-instrument diagnostic or Stock-Yogo effective F.",
            remedy="Interpret Wald confidence intervals cautiously; use weak-IV-robust tests.")
        diagnostics["Anderson-Rubin null beta=0"] = DiagnosticTest(
            name="Anderson-Rubin test (beta=0)", status=DiagnosticStatus.WARN,
            statistic_name=f"AR {ar_zero.distribution}", statistic_value=ar_zero.statistic,
            p_value=ar_zero.p_value, threshold="No certification of instrument validity",
            message="Weak-instrument-robust under the specified error assumptions; does not verify exclusion.",
            remedy="Interpret alongside instrument validity and alternative null hypotheses.")
        if g_count is not None and g_count < 30:
            diagnostics["Few-cluster approximation"] = DiagnosticTest(
                name="Few-cluster inference caution", status=DiagnosticStatus.WARN,
                statistic_name="Cluster count", statistic_value=float(g_count), p_value=None,
                threshold="Cluster-robust asymptotics require sufficient independent clusters",
                message="Cluster t/F approximations may be inaccurate with fewer than 30 clusters.")

        # A homoskedastic Sargan test is not calibrated for HC/CR inference.
        # Overidentification requires a separately validated robust test; do not
        # present an invalid p-value just to fill a reporting slot.
        sargan_p = None
        if k_inst > 1:
            diagnostics["Overidentification not assessed"] = DiagnosticTest(
                name="Overidentification (robust J pending)",
                status=DiagnosticStatus.WARN, statistic_name="Not available",
                statistic_value=0.0, p_value=None, threshold="Independent validation pending",
                message="The homoskedastic Sargan statistic is not valid for HC/CR inference; no overidentification p-value reported.")

        return IVResult(
            estimate=tau_estimate,
            se=se_estimate,
            p_value=p_val,
            ci=ci,
            outcome_name=self.outcome,
            endogenous_name=self.endogenous,
            instruments=self.instruments,
            first_stage_f=f_stat,
            sargan_j_p=sargan_p,
            n_obs=N,
            diagnostics=diagnostics,
            underlying_model=self,
            covariance=self.covariance, n_clusters=g_count,
            first_stage_robust=robust_first, first_stage_partial_r2=partial_r2,
            ar_null_zero=ar_zero,
        )