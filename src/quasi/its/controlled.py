"""Controlled Interrupted Time Series (CITS) multi-group segmented regression (Linden 2015).

Estimates:
    Y_gt = b0 + b1*T_t + b2*Z_g + b3*(Z_g*T_t) + b4*D_t + b5*P_t + b6*(Z_g*D_t) + b7*(Z_g*P_t) + e_gt
where Z_g = 1 for the treated series and 0 for the control series.
- b6: Difference-in-immediate-level-change (primary estimand)
- b7: Difference-in-post-intervention-slope-change
- b3: Pre-intervention trend difference (parallel pre-trends screen)
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from scipy import stats

from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.designs.validation import _regular_time
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.inference import bartlett_lag, ewc_har_covariance, hac_covariance, least_squares
from quasi.prep.period_index import SmartPeriodIndex


class CITSResult(QEDResult):
    """Result container for Controlled Interrupted Time Series (CITS)."""

    def __init__(
        self,
        *,
        level_diff: float,
        level_se: float,
        level_p: float,
        slope_diff: float,
        slope_se: float,
        slope_p: float,
        pre_trend_diff: float,
        pre_trend_p: float,
        outcome_name: str,
        n_obs: int,
        n_treated: int,
        time_labels: List[str],
        intervention_idx: int,
        y_treated: np.ndarray,
        y_control: np.ndarray,
        diagnostics: Dict[str, DiagnosticTest],
        alpha: float = 0.05,
        error_model: str = "ewc",
        degrees_freedom: Optional[int] = None,
        underlying_model: Optional[BaseQEDModel] = None,
    ) -> None:
        crit = (
            float(stats.t.ppf(1.0 - alpha / 2.0, df=degrees_freedom))
            if degrees_freedom is not None
            else float(stats.norm.ppf(1.0 - alpha / 2.0))
        )
        ci = (level_diff - crit * level_se, level_diff + crit * level_se)
        super().__init__(
            estimate=level_diff,
            se=level_se,
            p_value=level_p,
            ci=ci,
            estimand="Controlled ITS Difference-in-Level-Change (Treated vs Control)",
            outcome_name=outcome_name,
            model_name="ControlledITS",
            n_obs=n_obs,
            n_treated=n_treated,
            diagnostics=diagnostics,
            sub_effects={
                "level_diff": level_diff,
                "level_se": level_se,
                "level_p": level_p,
                "slope_diff": slope_diff,
                "slope_se": slope_se,
                "slope_p": slope_p,
                "pre_trend_diff": pre_trend_diff,
                "pre_trend_p": pre_trend_p,
            },
            model_meta={"error_model": error_model, "intervention_index": intervention_idx},
            underlying_model=underlying_model,
        )
        self.level_diff = level_diff
        self.level_se = level_se
        self.level_p = level_p
        self.slope_diff = slope_diff
        self.slope_se = slope_se
        self.slope_p = slope_p
        self.pre_trend_diff = pre_trend_diff
        self.pre_trend_p = pre_trend_p
        self.time_labels = time_labels
        self.intervention_idx = intervention_idx
        self.y_treated = y_treated
        self.y_control = y_control
        self.inference_details = {
            "method": "CITS_EWC_HAR_Fixed_b" if error_model == "ewc" else "CITS_HAC_Bartlett",
            "degrees_freedom": degrees_freedom,
            "reference_certified": False,
        }

    def summary(self) -> str:
        lines = [
            "[EXPERIMENTAL] Controlled Interrupted Time Series (Linden 2015)",
            "=" * 70,
            f"Outcome: {self.outcome_name} | Intervention: {self.time_labels[self.intervention_idx]}",
            f"Diff-in-Level-Change: {self.level_diff:+.4f} (SE: {self.level_se:.4f}, p={self.level_p:.4f})",
            f"{100 * self.confidence_level:g}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Diff-in-Slope-Change: {self.slope_diff:+.4f} (SE: {self.slope_se:.4f}, p={self.slope_p:.4f})",
            f"Pre-Trend Difference: {self.pre_trend_diff:+.4f} (p={self.pre_trend_p:.4f})",
            "=" * 70,
        ]
        return "\n".join(lines)

    def plot(self, **kwargs: Any):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=100)
        t_idx = np.arange(len(self.time_labels))
        ax.plot(t_idx, self.y_treated, color="#1B4F72", linewidth=2.2, marker="o", markersize=3, label="Treated Series")
        ax.plot(t_idx, self.y_control, color="#7F8C8D", linewidth=2.0, linestyle="--", marker="s", markersize=3, label="Control Series")
        ax.axvline(self.intervention_idx - 0.5, color="#C0392B", linestyle=":", linewidth=1.4, label="Intervention")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(self.outcome_name)
        ax.set_xlabel("Time Period")
        ax.set_title("Controlled Interrupted Time Series: Treated vs Control")
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig, ax

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
            f"A Controlled Interrupted Time Series (CITS) regression compared treated and control "
            f"trajectories for '{self.outcome_name}' with {self.inference_details['method']} inference. "
            f"The estimated difference-in-level-change is {self.level_diff:.4f} (SE={self.level_se:.4f}, p={self.level_p:.4f}) "
            f"and difference-in-slope-change is {self.slope_diff:.4f} (p={self.slope_p:.4f})."
        )


class ControlledITS(BaseQEDModel):
    """Two-group Controlled Interrupted Time Series on a balanced 2 x T panel."""

    def __init__(
        self,
        data: Any,
        outcome: str,
        time: str,
        group: str,
        intervention_date: Any,
        *,
        error_model: Literal["ewc", "hac"] = "ewc",
        hac_lag: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        if error_model not in ("ewc", "hac"):
            raise InvalidDesignError("ControlledITS error_model must be 'ewc' or 'hac'")
        if error_model == "ewc" and hac_lag is not None:
            raise InvalidDesignError("hac_lag is incompatible with error_model='ewc'")
        super().__init__(data=data, outcome=outcome, alpha=alpha)
        self.raw_data = data if isinstance(data, pd.DataFrame) else data.to_pandas()
        self.time_col = time
        self.group_col = group
        self.intervention_date = intervention_date
        self.error_model = error_model
        self.hac_lag = hac_lag

    def fit(self) -> CITSResult:
        df = self.raw_data.copy()
        for col in (self.outcome, self.time_col, self.group_col):
            if col not in df.columns:
                raise InvalidDesignError(f"Missing required column: {col}")
        if df[[self.outcome, self.time_col, self.group_col]].isna().any().any():
            raise InvalidDesignError("ControlledITS requires complete nonmissing data")
        if not df[self.group_col].isin([0, 1, False, True]).all():
            raise InvalidDesignError("ControlledITS group column must be binary 0 (control) / 1 (treated)")
        if df.duplicated([self.group_col, self.time_col]).any():
            raise InvalidDesignError("ControlledITS requires one observation per group-time")

        treated_df = df[df[self.group_col].astype(int) == 1].sort_values(self.time_col).reset_index(drop=True)
        control_df = df[df[self.group_col].astype(int) == 0].sort_values(self.time_col).reset_index(drop=True)
        if len(treated_df) != len(control_df) or not np.array_equal(
            treated_df[self.time_col].to_numpy(), control_df[self.time_col].to_numpy()
        ):
            raise InvalidDesignError("ControlledITS requires aligned time periods for treated and control series")
        if not _regular_time(treated_df[self.time_col]):
            raise InvalidDesignError("ControlledITS requires regular time spacing")

        period_index, _ = SmartPeriodIndex.from_series(treated_df[self.time_col])
        if self.intervention_date not in period_index.time_to_idx:
            raise InvalidDesignError("intervention_date not found in time column")
        interv_idx = period_index.time_to_idx[self.intervention_date]
        t_len = len(treated_df)
        if interv_idx < 4 or (t_len - interv_idx) < 3:
            raise InsufficientSupportError("ControlledITS requires >=4 pre and >=3 post periods per series")

        # Analyze the exact differenced series D_t = Y_{1t} - Y_{0t}:
        # Mathematically identical for (b2, b3, b6, b7) to the 8-parameter interacted
        # pooled regression, while keeping temporal ordering exact for HAR/HAC covariance!
        y_t = treated_df[self.outcome].to_numpy(dtype=float)
        y_c = control_df[self.outcome].to_numpy(dtype=float)
        y_diff = y_t - y_c
        t_seq = np.arange(t_len, dtype=float)
        d_post = (t_seq >= interv_idx).astype(float)
        p_post = np.maximum(0.0, t_seq - interv_idx)
        x_mat = np.column_stack([np.ones(t_len), t_seq, d_post, p_post])

        fit = least_squares(x_mat, y_diff)
        if self.error_model == "ewc":
            ewc_res = ewc_har_covariance(x_mat, fit.residuals)
            cov = ewc_res.covariance
            df_val: Optional[int] = ewc_res.degrees_freedom
            dist = stats.t(df=df_val)
        else:
            lag = bartlett_lag(t_len) if self.hac_lag is None else self.hac_lag
            cov = hac_covariance(x_mat, fit.residuals, max_lag=lag, small_sample=False)
            df_val = None
            dist = stats.norm

        diag_cov = np.diag(cov)
        if not np.isfinite(diag_cov).all() or np.any(diag_cov <= 0.0):
            raise InsufficientSupportError("ControlledITS covariance diagonal is nonpositive or nonfinite")
        ses = np.sqrt(diag_cov)
        pre_diff, pre_se = float(fit.coefficients[1]), float(ses[1])
        lvl_diff, lvl_se = float(fit.coefficients[2]), float(ses[2])
        slp_diff, slp_se = float(fit.coefficients[3]), float(ses[3])
        pre_p = float(2.0 * dist.sf(abs(pre_diff / pre_se)))
        lvl_p = float(2.0 * dist.sf(abs(lvl_diff / lvl_se)))
        slp_p = float(2.0 * dist.sf(abs(slp_diff / slp_se)))

        pre_status = DiagnosticStatus.PASS if pre_p >= 0.10 else (DiagnosticStatus.WARN if pre_p >= 0.05 else DiagnosticStatus.FAIL)
        diagnostics = {
            "Pre-Intervention Parallel Trends (CITS)": DiagnosticTest(
                name="Pre-intervention trend difference (Z * T)",
                status=pre_status,
                statistic_name="Pre-trend diff",
                statistic_value=pre_diff,
                p_value=pre_p,
                threshold="p >= 0.10 (descriptive screen)",
                message=(
                    "No statistically detectable pre-intervention slope divergence between treated and control series; "
                    "does not prove counterfactual parallelism."
                    if pre_p >= 0.10
                    else "Pre-intervention trends diverge between treated and control series."
                ),
            )
        }

        return CITSResult(
            level_diff=lvl_diff,
            level_se=lvl_se,
            level_p=lvl_p,
            slope_diff=slp_diff,
            slope_se=slp_se,
            slope_p=slp_p,
            pre_trend_diff=pre_diff,
            pre_trend_p=pre_p,
            outcome_name=self.outcome,
            n_obs=2 * t_len,
            n_treated=t_len - interv_idx,
            time_labels=period_index.labels,
            intervention_idx=interv_idx,
            y_treated=y_t,
            y_control=y_c,
            diagnostics=diagnostics,
            alpha=self.alpha,
            error_model=self.error_model,
            degrees_freedom=df_val,
            underlying_model=self,
        )
