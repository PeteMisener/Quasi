"""
Interrupted Time Series (ITS) segmented regression module.
Features Newey-West HAC standard error correction, Durbin-Watson tests,
and counterfactual extrapolation projections.
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
from quasi.prep.period_index import SmartPeriodIndex
from quasi.inference import (
    bartlett_lag,
    ewc_har_covariance,
    hac_covariance,
    least_squares,
    prewhitened_hac_covariance,
)
from quasi.inference.serial import prais_winsten_ar1, profile_ml_ar1, reml_ar1
from quasi.errors import InvalidDesignError, InsufficientSupportError


def compute_newey_west_hac(
    Z: np.ndarray,
    residuals: np.ndarray,
    max_lag: Optional[int] = None,
) -> np.ndarray:
    """
    Computes Newey-West (1987) Heteroskedasticity and Autocorrelation Consistent (HAC) covariance matrix.
    Uses Bartlett triangular kernel weights: w_l = 1 - l / (L + 1).
    """
    # Preserve public helper for backwards compatibility; fail on invalid lag
    # instead of silently clipping it. Explicit lag=0 means HC0 covariance.
    return hac_covariance(Z, residuals, max_lag=max_lag, small_sample=False)


class ITSResult(QEDResult):
    """Result container for Interrupted Time Series models."""

    def __init__(
        self,
        level_change: float,
        level_se: float,
        level_p: float,
        slope_change: float,
        slope_se: float,
        slope_p: float,
        pre_slope: float,
        outcome_name: str,
        n_obs: int,
        time_labels: List[str],
        intervention_idx: int,
        y_actual: np.ndarray,
        y_fitted: np.ndarray,
        y_counterfactual: np.ndarray,
        diagnostics: Dict[str, DiagnosticTest],
        alpha: float = 0.05,
        underlying_model: Optional[BaseQEDModel] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        error_model: str = "hac",
        degrees_freedom: Optional[int] = None,
        forecast_inputs: Optional[dict] = None,
        seasonal_period: Optional[int] = None,
        seasonal_harmonics: int = 1,
    ) -> None:
        # User-selected two-sided confidence interval (t for AR1/EWC fixed-b, normal for HAC).
        use_t_dist = error_model in ("ar1", "ar1_mle", "ar1_reml", "ewc") and degrees_freedom is not None
        crit = float(stats.t.ppf(1 - alpha / 2, df=degrees_freedom)) if use_t_dist else float(stats.norm.ppf(1 - alpha / 2))
        ci = (level_change - crit * level_se, level_change + crit * level_se)

        super().__init__(
            estimate=level_change,
            se=level_se,
            p_value=level_p,
            ci=ci,
            estimand="Immediate Level Change",
            outcome_name=outcome_name,
            model_name="InterruptedTimeSeries",
            n_obs=n_obs,
            n_treated=n_obs - intervention_idx,
            diagnostics=diagnostics,
            sub_effects={
                "immediate_level_change": level_change,
                "level_se": level_se,
                "level_p": level_p,
                "trend_slope_change": slope_change,
                "slope_se": slope_se,
                "slope_p": slope_p,
                "pre_intervention_slope": pre_slope,
            },
            model_meta={"intervention_index": intervention_idx, "error_model": error_model,
                        "seasonal_period": seasonal_period, "seasonal_harmonics": seasonal_harmonics if seasonal_period is not None else None},
            underlying_model=underlying_model,
        )
        self.level_change = level_change
        self.level_se = level_se
        self.level_p = level_p
        self.slope_change = slope_change
        self.slope_se = slope_se
        self.slope_p = slope_p
        self.pre_slope = pre_slope
        self.time_labels = time_labels
        self.intervention_idx = intervention_idx
        self.y_actual = y_actual
        self.y_fitted = y_fitted
        self.y_counterfactual = y_counterfactual
        self.error_model = error_model
        self._alpha = alpha
        self._forecast_inputs = forecast_inputs
        self.seasonal_period = seasonal_period
        self.seasonal_harmonics = seasonal_harmonics if seasonal_period is not None else None
        self._coefficient_covariance = np.asarray(covariance_matrix, dtype=float).copy() if covariance_matrix is not None else None
        self._critical_value = crit
        if self._coefficient_covariance is not None:
            # Sum of level+slope effect across ALL observed post periods.
            post_count = n_obs - intervention_idx
            contrast = np.array([0.0, 0.0, float(post_count),
                                 post_count*(post_count-1)/2], dtype=float)
            aggregate = float(post_count * level_change + contrast[3] * slope_change)
            aggregate_var = float(contrast @ self._coefficient_covariance[:4, :4] @ contrast)
            if not np.isfinite(aggregate_var) or aggregate_var < -1e-10:
                raise InsufficientSupportError("Invalid cumulative ITS effect covariance")
            aggregate_se = float(np.sqrt(max(aggregate_var, 0.0)))
            self.sub_effects.update({
                "cumulative_post_effect": aggregate,
                "cumulative_post_se": aggregate_se,
                "cumulative_post_ci_lower": aggregate - crit * aggregate_se,
                "cumulative_post_ci_upper": aggregate + crit * aggregate_se,
                "average_post_effect": aggregate / post_count,
                "average_post_se": aggregate_se / post_count,
            })
        _ar_models = ("ar1", "ar1_mle", "ar1_reml")
        _method_map = {
            "hac": "OLS_HAC_Bartlett",
            "hac_pw": "OLS_HAC_Andrews_Monahan_Prewhitened",
            "ewc": "OLS_HAR_EWC_Fixed_b",
            "ar1": "Prais_Winsten_FGLS_AR1",
            "ar1_mle": "AR1_stationary_Gaussian_profile_ML",
            "ar1_reml": "AR1_stationary_Gaussian_REML",
        }
        _cov_map = {
            "hac": "HAC_Bartlett",
            "hac_pw": "HAC_Prewhitened_Andrews",
            "ewc": "HAR_EWC_Fixed_b",
            "ar1": "AR1_model_based",
            "ar1_mle": "AR1_model_based",
            "ar1_reml": "AR1_model_based",
        }
        self.inference_details = {
            "method": _method_map.get(error_model, "OLS_HAC_Bartlett"),
            "covariance": _cov_map.get(error_model, "HAC_Bartlett"),
            "max_lag": getattr(underlying_model, "_effective_hac_lag", None) if error_model == "hac" else None,
            "n_cosines": getattr(underlying_model, "_ewc_cosines", None) if error_model == "ewc" else None,
            "estimated_rho": getattr(underlying_model, "_estimated_ar1_rho", None) if error_model in _ar_models else None,
            "iterations": getattr(underlying_model, "_ar1_iterations", None) if error_model in _ar_models else None,
            "degrees_freedom": degrees_freedom,
            "small_sample": error_model in (*_ar_models, "ewc"),
            "rho_estimation_uncertainty_accounted_for": False if error_model in _ar_models else None,
            "variance_assumption": (
                "stationary_Gaussian_AR1" if error_model in _ar_models
                else ("fixed_b_HAR_EWC" if error_model == "ewc" else "HAC_asymptotic")
            ),
            "conditional_on_estimated_rho": error_model in _ar_models,
            "reference_certified": False,
            "seasonal_period": seasonal_period,
            "seasonal_harmonics": seasonal_harmonics if seasonal_period is not None else None,
        }

    def effect_trajectory(self) -> pd.DataFrame:
        """Post-intervention fitted mean contrasts and correlated cumulative uncertainty.

        These conditional mean contrasts exclude future innovations, uncertainty
        about counterfactual functional form, and estimation of the AR parameter.
        """
        if self._coefficient_covariance is None:
            raise InvalidDesignError("Coefficient covariance is not available")
        n_post = self.n_obs - self.intervention_idx
        post_index = np.arange(n_post, dtype=float)
        contrasts = np.column_stack((np.zeros(n_post), np.zeros(n_post),
                                     np.ones(n_post), post_index))
        var = np.einsum("ij,jk,ik->i", contrasts, self._coefficient_covariance[:4, :4], contrasts)
        cumulative = np.cumsum(contrasts, axis=0)
        cum_var = np.einsum("ij,jk,ik->i", cumulative, self._coefficient_covariance[:4, :4], cumulative)
        if np.any(var < -1e-10) or np.any(cum_var < -1e-10):
            raise InsufficientSupportError("Negative effect variance from fitted covariance")
        effects = self.level_change + self.slope_change * post_index
        totals = np.cumsum(effects)
        se = np.sqrt(np.maximum(var, 0))
        cum_se = np.sqrt(np.maximum(cum_var, 0))
        c = self._critical_value
        return pd.DataFrame({
            "time": self.time_labels[self.intervention_idx:],
            "post_period": post_index.astype(int),
            "effect": effects, "se": se,
            "ci_lower": effects-c*se, "ci_upper": effects+c*se,
            "cumulative_effect": totals, "cumulative_se": cum_se,
            "cumulative_ci_lower": totals-c*cum_se,
            "cumulative_ci_upper": totals+c*cum_se,
        })

    def counterfactual_prediction(self) -> pd.DataFrame:
        """Retrospective AR(1) conditional *outcome prediction*, not a causal CI.

        Given the last preintervention residual, the post-period no-intervention
        path is propagated using rho. Intervals combine AR(1) future innovation
        variance with the *conditional-on-rho* fitted coefficient covariance.
        Rho estimation, specification/model-selection uncertainty and the fact
        that the regression was fitted using postintervention outcomes are not
        accounted for. The postintervention covariate path is assumed known.
        """
        if self.error_model not in ("ar1", "ar1_mle", "ar1_reml") or self._forecast_inputs is None:
            raise InvalidDesignError("AR1 model required for counterfactual outcome prediction")
        info = self._forecast_inputs
        xcf = info["counterfactual_design"]
        xpre = info["last_pre_design"]
        beta = info["coefficients"]
        ypre = float(info["last_pre_y"])
        rho = float(info["rho"])
        sigma2 = float(info["innovation_variance"])
        cov_beta = self._coefficient_covariance
        npost = len(xcf)
        steps = np.arange(1, npost+1, dtype=float)
        decays = rho ** steps
        # The preintervention residual depends on beta: d/d beta of the
        # conditional forecast is xcf[k] - rho**(k+1) * x_pre.
        effective_x = xcf - decays[:, None] * xpre[None, :]
        mean = xcf @ beta + decays * (ypre - xpre @ beta)
        mean_var = np.einsum("ij,jk,ik->i", effective_x, cov_beta, effective_x)
        cumulative_x = np.cumsum(effective_x, axis=0)
        cumulative_mean_var = np.einsum("ij,jk,ik->i", cumulative_x, cov_beta, cumulative_x)
        # Innovations satisfy e_k=rho*e_(k-1)+u_k, S_k=S_(k-1)+e_k.
        # Propagate exact conditional variance without building an N x N matrix.
        v_e = 0.0
        v_s = 0.0
        cov_se = 0.0
        innovation_var = np.empty(npost)
        cum_innovation_var = np.empty(npost)
        for k in range(npost):
            cov_previous_s_new_e = rho * cov_se
            v_e = rho*rho*v_e + sigma2
            v_s = v_s + 2*cov_previous_s_new_e + v_e
            cov_se = cov_previous_s_new_e + v_e
            innovation_var[k] = v_e
            cum_innovation_var[k] = v_s
        total_var = mean_var + innovation_var
        cumulative_var = cumulative_mean_var + cum_innovation_var
        if not np.isfinite(total_var).all() or not np.isfinite(cumulative_var).all() or np.min(total_var) <= 0:
            raise InsufficientSupportError("Invalid conditional counterfactual prediction variance")
        # Approximate two-sided t critical value, conditional on fitted rho.
        c = self._critical_value
        se = np.sqrt(np.maximum(total_var, 0))
        cse = np.sqrt(np.maximum(cumulative_var, 0))
        cs = np.cumsum(mean)
        return pd.DataFrame({
            "time": self.time_labels[self.intervention_idx:],
            "post_period": (steps-1).astype(int),
            "counterfactual_mean": mean,
            "prediction_se": se,
            "prediction_lower": mean-c*se,
            "prediction_upper": mean+c*se,
            "cumulative_counterfactual_mean": cs,
            "cumulative_prediction_se": cse,
            "cumulative_prediction_lower": cs-c*cse,
            "cumulative_prediction_upper": cs+c*cse,
        })

    def ar1_bootstrap_level_interval(self, *, reps: int = 399, seed: int = 42) -> dict:
        """Exploratory Gaussian parametric *basic bootstrap* interval for the level.

        Refit the selected AR(1) method (including rho) on each parametric sample under the fitted
        segmented model and stationary AR(1) disturbance distribution. This
        includes some rho-estimation variability, but still relies on the
        assumed correctly specified Gaussian AR(1) DGP, fixed intervention
        timing and known design/covariates. It is NOT generally coverage-valid.
        Bootstrap failures are counted, and >5% failures prevent reporting.
        """
        if self.error_model not in ("ar1", "ar1_mle", "ar1_reml") or self._forecast_inputs is None:
            raise InvalidDesignError("AR1 model required for parametric bootstrap")
        if type(reps) is not int or reps < 50 or reps > 20000:
            raise InvalidDesignError("reps must be an integer in [50, 20000]")
        if type(seed) is not int or seed < 0:
            raise InvalidDesignError("seed must be a nonnegative integer")
        info = self._forecast_inputs
        x = info["full_design"]
        mu = x @ info["coefficients"]
        n = len(mu)
        rho = float(info["rho"])
        sigma = float(np.sqrt(info["innovation_variance"]))
        rng = np.random.default_rng(seed)
        fitted_betas = []
        fitted_rhos = []
        failures = 0
        for _ in range(reps):
            u = rng.normal(0, sigma, n)
            eps = np.empty(n, dtype=float)
            eps[0] = u[0] / np.sqrt(1-rho*rho)
            for j in range(1,n):
                eps[j] = rho*eps[j-1] + u[j]
            try:
                fitted = (
                    reml_ar1(x, mu + eps, tolerance=info["tolerance"])
                    if self.error_model == "ar1_reml"
                    else (
                        profile_ml_ar1(x, mu + eps, tolerance=info["tolerance"])
                        if self.error_model == "ar1_mle"
                        else prais_winsten_ar1(
                            x, mu + eps, max_iter=info["max_iter"], tolerance=info["tolerance"]
                        )
                    )
                )
            except InsufficientSupportError:
                failures += 1
                continue
            fitted_betas.append(float(fitted.coefficients[2]))
            fitted_rhos.append(float(fitted.rho))
        if len(fitted_betas) < np.ceil(0.95 * reps):
            raise InsufficientSupportError(
                f"Bootstrap failed in {failures}/{reps} refits; refusing an interval")
        # Basic/bootstrap pivot intervals, not the percentile interval.
        lo, hi = np.quantile(fitted_betas, [self._alpha/2, 1-self._alpha/2])
        interval = (float(2*self.level_change-hi), float(2*self.level_change-lo))
        return {
            "estimate": self.level_change,
            "ci_lower": interval[0], "ci_upper": interval[1],
            "confidence_level": self.confidence_level,
            "method": "experimental_Gaussian_AR1_basic_parametric_bootstrap",
            "replications": reps, "successful_refits": len(fitted_betas),
            "failed_refits": failures, "seed": seed,
            "rho_quantiles_05_50_95": [float(v) for v in np.quantile(fitted_rhos, [.05,.5,.95])],
            "rho_reestimated_each_replication": True,
            "error_model": self.error_model,
            "reference_certified": False,
            "limitations": "Assumes stationary Gaussian AR1, fixed intervention date and design, and plug-in DGP; not proven finite-sample calibrated",
        }

    def in_time_placebo(self, placebo_dates: Optional[List[Any]] = None) -> pd.DataFrame:
        """Pre-intervention falsification sweep using strictly pre-intervention data."""
        if self._model is None:
            raise InvalidDesignError("Underlying model required for in-time placebo tests")
        df_sorted = self._model.raw_data.copy().sort_values(self._model.time_col).reset_index(drop=True)
        pre_df = df_sorted.iloc[: self.intervention_idx].copy()
        n_pre = len(pre_df)
        if n_pre < 8:
            raise InsufficientSupportError("At least 8 pre-intervention periods are required for in-time placebos")
        if placebo_dates is None:
            # Default: quartiles within the pre-intervention window leaving >=3 obs on each side
            cand_idx = sorted({max(3, n_pre // 3), n_pre // 2, min(n_pre - 3, (2 * n_pre) // 3)})
            dates = [pre_df[self._model.time_col].iloc[i] for i in cand_idx if 3 <= i <= n_pre - 3]
        else:
            dates = list(placebo_dates)
        rows = []
        for dt in dates:
            fit = InterruptedTimeSeries(
                pre_df,
                outcome=self.outcome_name,
                time=self._model.time_col,
                intervention_date=dt,
                covariates=self._model.covariates or None,
                hac_lag=self._model.hac_lag,
                alpha=self._alpha,
                error_model=self.error_model,
                ar1_max_iter=self._model.ar1_max_iter,
                ar1_tolerance=self._model.ar1_tolerance,
            ).fit()
            rows.append({
                "placebo_date": dt,
                "level_change": fit.level_change,
                "level_se": fit.level_se,
                "level_p": fit.level_p,
                "slope_change": fit.slope_change,
                "slope_se": fit.slope_se,
                "slope_p": fit.slope_p,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        interv_label = self.time_labels[self.intervention_idx]
        lines = [
            "[EXPERIMENTAL] Research prototype; reference validation pending.",
            "=" * 70,
            f"Interrupted Time Series | Outcome: {self.outcome_name} (Intervention: {interv_label})",
            "=" * 70,
            f"Pre-Intervention Baseline Trend:  {self.pre_slope:+.4f}",
            "-" * 70,
            f"Immediate Level Change (Step):    {self.level_change:+.4f} ({self.error_model.upper()} SE: {self.level_se:.4f}, p = {self.level_p:.4f})",
            f"{100*self.confidence_level:.0f}% Level CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Trend Slope Change (Slope):       {self.slope_change:+.4f} ({self.error_model.upper()} SE: {self.slope_se:.4f}, p = {self.slope_p:.4f})",
            f"Total Observed Periods:           {self.n_obs} ({self.intervention_idx} Pre, {self.n_obs - self.intervention_idx} Post)",
            "-" * 70,
            "Autocorrelation & Specification Diagnostics:",
        ]
        for t in self.diagnostics.values():
            p_str = f"p = {t.p_value:.4f}" if t.p_value is not None else "-"
            lines.append(f"  • {t.name:<32} [{t.status.value}] {t.statistic_name} = {t.statistic_value:.4f} ({p_str})")
        lines.append("=" * 70)
        return "\n".join(lines)

    def plot(self, **kwargs: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        from quasi.visualization.its_plots import render_its_figure

        return render_its_figure(
            time_labels=self.time_labels,
            y_actual=self.y_actual,
            y_fitted=self.y_fitted,
            y_counterfactual=self.y_counterfactual,
            intervention_idx=self.intervention_idx,
            outcome_name=self.outcome_name,
            **kwargs,
        )

    def bibtex(self) -> str:
        return """@article{linden2015conducting,
  title={Conducting interrupted time-series analysis for single-and multiple-group comparisons},
  author={Linden, Ariel},
  journal={The Stata Journal},
  volume={15},
  number={2},
  pages={480--500},
  year={2015}
}"""

    def methods_text(self) -> str:
        return (
            f"The intervention effect on '{self.outcome_name}' was estimated using segmented interrupted "
            f"time series regression. Inference uses {self.inference_details['method']} under its "
            f"method-specific assumptions. The fitted model estimates "
            f"an immediate level change of {self.level_change:.3f} (p = {self.level_p:.4f}) and a post-intervention "
            f"slope change of {self.slope_change:.3f} per period (p = {self.slope_p:.4f})."
        )


class InterruptedTimeSeries(BaseQEDModel):
    """
    Segmented Linear Interrupted Time Series Estimator.
    Estimates: Y_t = beta_0 + beta_1 * T_t + beta_2 * D_t + beta_3 * P_t + X_t * gamma + eps_t

    Parameters:
        data: DataFrame containing time series.
        outcome: Dependent variable column name.
        time: Temporal order column name.
        intervention_date: Value in `time` column denoting the first post-intervention period.
        covariates: Optional exogenous continuous confounders.
        hac_lag: Max lag for Newey-West HAC covariance. Default uses automated rule.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, Any],
        outcome: str,
        time: str,
        intervention_date: Any,
        covariates: Optional[List[str]] = None,
        hac_lag: Optional[int] = None,
        alpha: float = 0.05,
        error_model: str = "hac",
        ar1_max_iter: int = 100,
        ar1_tolerance: float = 1e-7,
        seasonal_period: Optional[int] = None,
        seasonal_harmonics: int = 1,
    ) -> None:
        super().__init__(data=data, outcome=outcome, covariates=covariates, alpha=alpha)
        self.raw_data = data if isinstance(data, pd.DataFrame) else data.to_pandas()
        self.time_col = time
        self.intervention_date = intervention_date
        if error_model not in ("hac", "hac_pw", "ewc", "ar1", "ar1_mle", "ar1_reml"):
            raise InvalidDesignError("error_model must be 'hac', 'hac_pw', 'ewc', 'ar1', 'ar1_mle', or 'ar1_reml'")
        if error_model in ("ar1", "ar1_mle", "ar1_reml", "ewc") and hac_lag is not None:
            raise InvalidDesignError("hac_lag is incompatible with AR1 or EWC error models")
        if error_model in ("ar1_mle", "ar1_reml") and ar1_max_iter != 100:
            raise InvalidDesignError("ar1_max_iter applies only to error_model='ar1'; use ar1_tolerance for ar1_mle/ar1_reml")
        if type(ar1_max_iter) is not int or not 2 <= ar1_max_iter <= 10000:
            raise InvalidDesignError("ar1_max_iter must be an integer between 2 and 10000")
        if isinstance(ar1_tolerance, bool) or not isinstance(ar1_tolerance, (int, float)) or not np.isfinite(ar1_tolerance) or not 0 < ar1_tolerance < 0.1:
            raise InvalidDesignError("ar1_tolerance must be finite and between zero and 0.1")
        if seasonal_period is None:
            if seasonal_harmonics != 1:
                raise InvalidDesignError("seasonal_harmonics requires a seasonal_period")
        else:
            if type(seasonal_period) is not int or seasonal_period < 2:
                raise InvalidDesignError("seasonal_period must be an integer >= 2")
            if type(seasonal_harmonics) is not int or not 1 <= seasonal_harmonics <= seasonal_period//2:
                raise InvalidDesignError("seasonal_harmonics must be between 1 and floor(period/2)")
        self.seasonal_period = seasonal_period
        self.seasonal_harmonics = seasonal_harmonics
        self.hac_lag = hac_lag
        self.error_model = error_model
        self.ar1_max_iter = ar1_max_iter
        self.ar1_tolerance = ar1_tolerance

    def fit(self) -> ITSResult:
        df = self.raw_data.copy().sort_values(self.time_col).reset_index(drop=True)
        if df[self.time_col].isna().any() or df[self.time_col].duplicated().any():
            raise ValueError("Interrupted Time Series requires unique nonmissing time points; duplicate times are unsupported.")

        # Direct estimator constructors must enforce the same cadence checks as
        # the explicit design validation path: positional indices are NOT time.
        from quasi.designs.validation import _regular_time
        if not _regular_time(df[self.time_col]):
            raise InvalidDesignError("ITS requires regular time spacing; gaps are unsupported")
        period_index, time_int = SmartPeriodIndex.from_series(df[self.time_col])
        df["_t"] = time_int

        if self.intervention_date not in period_index.time_to_idx:
            raise ValueError(f"Intervention date '{self.intervention_date}' not found in time series.")
        interv_idx = period_index.time_to_idx[self.intervention_date]

        T = len(df)
        if interv_idx < 3 or (T - interv_idx) < 3:
            raise ValueError("Interrupted Time Series requires at least 3 pre- and 3 post-intervention observations.")

        if self.seasonal_period is not None and (interv_idx < 2*self.seasonal_period or T-interv_idx < self.seasonal_period):
            raise InsufficientSupportError("Seasonal ITS requires at least 2 full preintervention cycles and 1 postintervention cycle")

        Y = df[self.outcome].to_numpy(dtype=float)

        # Construct segmented regressors:
        # T_t: 0, 1, 2, ..., T-1
        # D_t: 1 if t >= interv_idx else 0
        # P_t: (t - interv_idx) if t >= interv_idx else 0
        t_seq = np.arange(T, dtype=float)
        d_post = (t_seq >= interv_idx).astype(float)
        p_post = np.maximum(0.0, t_seq - interv_idx)

        regressor_cols = [np.ones(T), t_seq, d_post, p_post]
        if self.covariates:
            for cov in self.covariates:
                regressor_cols.append(df[cov].to_numpy(dtype=float))
        if self.seasonal_period is not None:
            # Fixed, prespecified Fourier components. For an even period,
            # the Nyquist sine is identically zero and must be omitted.
            for k in range(1, self.seasonal_harmonics + 1):
                angle = 2*np.pi*k*t_seq/self.seasonal_period
                regressor_cols.append(np.cos(angle))
                if 2*k != self.seasonal_period:
                    regressor_cols.append(np.sin(angle))

        Z = np.column_stack(regressor_cols)

        if self.error_model in ("ar1", "ar1_mle", "ar1_reml"):
            ar = (
                reml_ar1(Z, Y, tolerance=self.ar1_tolerance)
                if self.error_model == "ar1_reml"
                else (
                    profile_ml_ar1(Z, Y, tolerance=self.ar1_tolerance)
                    if self.error_model == "ar1_mle"
                    else prais_winsten_ar1(
                        Z, Y, max_iter=self.ar1_max_iter, tolerance=self.ar1_tolerance
                    )
                )
            )
            beta, fitted, residuals = ar.coefficients, ar.fitted, ar.residuals
            coefficient_covariance = ar.covariance
            self._estimated_ar1_rho = ar.rho
            self._ar1_iterations = ar.iterations
            self._effective_hac_lag = None
            self._ewc_cosines = None
            critical_dist = stats.t(df=ar.degrees_freedom)
            degrees_freedom = ar.degrees_freedom
        elif self.error_model == "ewc":
            fit = least_squares(Z, Y)
            beta, fitted, residuals = fit.coefficients, fit.fitted, fit.residuals
            ewc_fit = ewc_har_covariance(Z, residuals)
            coefficient_covariance = ewc_fit.covariance
            self._effective_hac_lag = None
            self._ewc_cosines = ewc_fit.n_cosines
            critical_dist = stats.t(df=ewc_fit.degrees_freedom)
            degrees_freedom = ewc_fit.degrees_freedom
        elif self.error_model == "hac_pw":
            fit = least_squares(Z, Y)
            beta, fitted, residuals = fit.coefficients, fit.fitted, fit.residuals
            coefficient_covariance = prewhitened_hac_covariance(
                Z, residuals, kernel="bartlett", max_lag=self.hac_lag, prewhiten=True
            )
            self._effective_hac_lag = self.hac_lag
            self._ewc_cosines = None
            critical_dist = stats.norm
            degrees_freedom = None
        else:
            fit = least_squares(Z, Y)
            beta, fitted, residuals = fit.coefficients, fit.fitted, fit.residuals
            self._effective_hac_lag = bartlett_lag(T) if self.hac_lag is None else self.hac_lag
            self._ewc_cosines = None
            coefficient_covariance = compute_newey_west_hac(Z, residuals, max_lag=self._effective_hac_lag)
            critical_dist = stats.norm
            degrees_freedom = None
        diag = np.diag(coefficient_covariance)
        if not np.isfinite(diag).all() or np.any(diag <= 0):
            raise InsufficientSupportError("ITS coefficient variance is nonpositive; no reliable interval can be reported")
        ses = np.sqrt(diag)
        level_est, level_se = float(beta[2]), float(ses[2])
        level_p = float(2.0 * critical_dist.sf(abs(level_est / level_se)))
        slope_est, slope_se = float(beta[3]), float(ses[3])
        slope_p = float(2.0 * critical_dist.sf(abs(slope_est / slope_se)))

        pre_slope = float(beta[1])

        # Counterfactual projection: D_t = 0, P_t = 0
        Z_cf = Z.copy()
        Z_cf[:, 2] = 0.0  # Nullify level step
        Z_cf[:, 3] = 0.0  # Nullify slope change
        counterfactual = Z_cf @ beta

        # For AR(1) FGLS, raw residual correlation is EXPECTED. Examine the
        # whitened innovations instead of incorrectly diagnosing that modeled
        # correlation as a failure. DW is descriptive, not a formal p-value.
        if self.error_model in ("ar1", "ar1_mle", "ar1_reml"):
            rho = self._estimated_ar1_rho
            checked_resid = residuals.copy()
            checked_resid[0] *= np.sqrt(1 - rho*rho)
            checked_resid[1:] = residuals[1:] - rho*residuals[:-1]
            diag_label = "Durbin-Watson AR1 innovations"
        else:
            checked_resid = residuals
            diag_label = "Durbin-Watson Autocorrelation"
        diff_res = np.diff(checked_resid)
        dw_stat = float(np.sum(diff_res**2) / np.maximum(np.sum(checked_resid**2), 1e-10))
        dw_status = DiagnosticStatus.PASS if (1.5 <= dw_stat <= 2.5) else DiagnosticStatus.WARN
        diagnostics: Dict[str, DiagnosticTest] = {
            diag_label: DiagnosticTest(
                name="Durbin-Watson innovation correlation" if self.error_model in ("ar1", "ar1_mle", "ar1_reml") else "Durbin-Watson residual correlation",
                status=dw_status,
                statistic_name="DW Stat", statistic_value=dw_stat, p_value=None,
                threshold="descriptive heuristic: 1.5 <= DW <= 2.5",
                message="No strong lag-one serial dependence indicated by this heuristic; not an assumption proof"
                if dw_status == DiagnosticStatus.PASS else
                f"Remaining serial dependence indicated (DW = {dw_stat:.2f}); review model assumptions and interval calibration",
                remedy="Consider error-model misspecification, seasonality, and independent reference tests"
                if dw_status == DiagnosticStatus.WARN else None,
            )
        }

        # Descriptive Ljung-Box innovation/residual diagnostic at multiple
        # lags. The chi-square reference is approximate because regression
        # coefficients and (for PW) rho were fitted; it is NOT a validity test.
        max_q_lag = min(8, max(2, T//5))
        centered = checked_resid - np.mean(checked_resid)
        denom = float(centered @ centered)
        if denom > 1e-12:
            lag_corrs = [float(np.dot(centered[lag:], centered[:-lag])/denom)
                         for lag in range(1, max_q_lag+1)]
            q = float(T*(T+2)*sum(v*v/(T-lag) for lag,v in enumerate(lag_corrs, start=1)))
            # Use conservative plain lag count rather than claiming a
            # regression-adjusted calibrated p-value for the fitted model.
            p_approx = float(stats.chi2.sf(q, max_q_lag))
            status = DiagnosticStatus.WARN if p_approx < 0.05 else DiagnosticStatus.PASS
            diagnostics["Ljung-Box residual screen"] = DiagnosticTest(
                name="Ljung-Box residual screen (descriptive)", status=status,
                statistic_name=f"Q({max_q_lag})", statistic_value=q,
                p_value=p_approx, threshold="exploratory chi-square screen; not calibrated after AR estimation",
                message="Residual serial dependence may remain; inspect seasonal or higher-order error models"
                if status == DiagnosticStatus.WARN else
                "No detected residual correlation at selected lags; not proof of independence",
                remedy="Check seasonal and higher-order autocorrelation; consider model comparison" if status == DiagnosticStatus.WARN else None,
            )

        # Explicitly distinguish effect-of-intervention *mean contrasts* from
        # no-intervention outcome *prediction intervals*. The latter use the
        # last pre-intervention residual and model-based AR1 innovation risk.
        forecast_inputs = None
        if self.error_model in ("ar1", "ar1_mle", "ar1_reml"):
            forecast_inputs = {
                "full_design": Z.copy(),
                "max_iter": self.ar1_max_iter,
                "tolerance": self.ar1_tolerance,
                "counterfactual_design": Z_cf[interv_idx:].copy(),
                "last_pre_design": Z[interv_idx-1].copy(),
                "last_pre_y": float(Y[interv_idx-1]),
                "coefficients": beta.copy(),
                "rho": float(ar.rho),
                "innovation_variance": float(ar.innovation_variance),
            }

        return ITSResult(
            level_change=level_est,
            level_se=level_se,
            level_p=level_p,
            slope_change=slope_est,
            slope_se=slope_se,
            slope_p=slope_p,
            pre_slope=pre_slope,
            outcome_name=self.outcome,
            n_obs=T,
            time_labels=period_index.labels,
            intervention_idx=interv_idx,
            y_actual=Y,
            y_fitted=fitted,
            y_counterfactual=counterfactual,
            diagnostics=diagnostics,
            alpha=self.alpha,
            underlying_model=self,
            covariance_matrix=coefficient_covariance,
            error_model=self.error_model,
            degrees_freedom=degrees_freedom,
            forecast_inputs=forecast_inputs,
            seasonal_period=self.seasonal_period,
            seasonal_harmonics=self.seasonal_harmonics,
        )