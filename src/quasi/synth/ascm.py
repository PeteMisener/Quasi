"""Ridge-Augmented Synthetic Control Method (Ben-Michael, Feller & Rothstein 2021, JASA).

Debiases classic simplex SCM when pre-treatment fit is imperfect by combining
the simplex SCM weights with a ridge outcome-model correction, with GCV/LOO
selection of the ridge penalty lambda.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.errors import InvalidDesignError
from quasi.inference.optimization import solve_simplex_regression
from quasi.synth.conformal import ConformalIntervalResult, conformal_sc_interval
from quasi.synth.panel import prepare_synthetic_panel


def _fit_ascm_weights(
    target_pre: np.ndarray,
    donor_pre: np.ndarray,
    *,
    ridge_lambda: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute (w_ascm, w_scm, chosen_lambda, intercept) per Ben-Michael et al. (2021) Section 4.1.

    Uses intercept-shifted (unit-pre-mean-demeaned) simplex SCM plus donor-centered
    ridge debiasing so:
    1. sum(w_ascm) == 1 identically,
    2. estimates are invariant to arbitrary global shifts Y -> Y + c, and
    3. constant level offsets between treated and donor units are absorbed cleanly.
    """
    n0, t0 = donor_pre.shape
    t1_mean = float(np.mean(target_pre))
    d_means = np.mean(donor_pre, axis=1)  # (N0,)
    target_dm = target_pre - t1_mean
    donor_dm = donor_pre - d_means[:, None]

    w_scm = solve_simplex_regression(donor_dm.T, target_dm, ridge=0.0, intercept=False).weights
    # Center pre-treatment columns across donors so 1' delta_w == 0 and sum(w_ascm) == 1 identically.
    x0 = donor_dm - donor_dm.mean(axis=0, keepdims=True)  # N0 x T0
    residual_scm = target_dm - donor_dm.T @ w_scm  # T0

    # SVD of centered X0 (N0 x T0) for closed-form GCV ridge selection
    u, s, vt = np.linalg.svd(x0, full_matrices=False)
    s2 = s * s

    if ridge_lambda is not None:
        if isinstance(ridge_lambda, bool) or not isinstance(ridge_lambda, (int, float)) or ridge_lambda <= 0 or not np.isfinite(ridge_lambda):
            raise InvalidDesignError("ridge_lambda must be strictly positive and finite")
        lam = float(ridge_lambda)
    else:
        # GCV grid over spectrum of X0'X0, skipping near-interpolating candidates (1 - df/T0 <= 0.05)
        scale_s2 = max(float(np.max(s2)), 1e-12)
        grid = scale_s2 * np.logspace(-3, 2, 40)
        best_lam = float(grid[-1])
        best_gcv = float("inf")
        proj = vt @ residual_scm
        for cand in grid:
            shrink = s2 / (s2 + cand)
            df_eff = float(np.sum(shrink))
            dof_frac = 1.0 - df_eff / t0
            if dof_frac <= 0.05:
                continue
            fitted = vt.T @ (shrink * proj)
            err = residual_scm - fitted
            gcv = float(np.mean(err * err) / (dof_frac * dof_frac))
            if gcv < best_gcv:
                best_gcv = gcv
                best_lam = float(cand)
        lam = best_lam

    # Ridge-augmented weights: w_ascm = w_scm + X0_c (X0_c' X0_c + lam I)^{-1} residual_scm
    delta_w = u @ ((s / (s2 + lam)) * (vt @ residual_scm))
    w_ascm = w_scm + delta_w
    intercept = float(t1_mean - float(w_ascm @ d_means))
    return w_ascm, w_scm, lam, intercept


class ASCMResult(QEDResult):
    """Result container for Ridge-Augmented Synthetic Control."""

    def __init__(
        self,
        *,
        estimate: float,
        outcome: str,
        unit_name: str,
        n_obs: int,
        weights_ascm: dict[str, float],
        weights_scm: dict[str, float],
        ridge_lambda: float,
        actual: np.ndarray,
        synthetic: np.ndarray,
        time_labels: tuple[str, ...],
        split: int,
        treatment_audit: dict[str, Any],
        model: BaseQEDModel,
        donors: np.ndarray,
    ) -> None:
        pre_rmspe = float(np.sqrt(np.mean((actual[:split] - synthetic[:split]) ** 2)))
        post_rmspe = float(np.sqrt(np.mean((actual[split:] - synthetic[split:]) ** 2)))
        diag = DiagnosticTest(
            name="ASCM Pre-Treatment RMSPE (descriptive)",
            status=DiagnosticStatus.WARN,
            statistic_name="RMSPE",
            statistic_value=pre_rmspe,
            p_value=None,
            threshold="descriptive fit only",
            message="Ridge debiasing improves pre-treatment balance via extrapolation; check negative weights.",
        )
        super().__init__(
            estimate=estimate,
            se=float("nan"),
            p_value=float("nan"),
            ci=(float("nan"), float("nan")),
            estimand="Post-period average Ridge-Augmented SCM gap",
            outcome_name=outcome,
            model_name="AugmentedSCM",
            n_obs=n_obs,
            n_treated=1,
            diagnostics={"pre_rmspe": diag},
            sub_effects={
                "treated_unit": unit_name,
                "donor_weights": weights_ascm,
                "scm_weights": weights_scm,
                "ridge_lambda": ridge_lambda,
                "pre_rmspe": pre_rmspe,
                "post_rmspe": post_rmspe,
                "l2_weight_norm": float(np.sqrt(sum(v * v for v in weights_ascm.values()))),
                "min_weight": float(min(weights_ascm.values())),
            },
            model_meta={"n_donors": len(weights_ascm), "ridge_lambda": ridge_lambda, "treatment_audit": treatment_audit},
            underlying_model=model,
        )
        self.inference_details = {
            "method": "not_available",
            "reference_certified": False,
            "reason": "Use conformal_interval() for finite-sample conformal bounds.",
        }
        self.donor_weights = weights_ascm
        self.scm_weights = weights_scm
        self.ridge_lambda = ridge_lambda
        self.actual_trajectory = actual.copy()
        self.synthetic_trajectory = synthetic.copy()
        self.pre_rmspe = pre_rmspe
        self.post_rmspe = post_rmspe
        self._time_labels = list(time_labels)
        self._split = split
        self._donors = donors.copy()

    def to_metadata_dict(self) -> dict[str, Any]:
        rec = super().to_metadata_dict()
        rec["primary_metric"].update({"se": None, "p_value": None, "ci": (None, None)})
        rec["primary_metric"].pop("ci_95", None)
        return rec

    def conformal_interval(self, *, alpha: float = 0.05) -> ConformalIntervalResult:
        """Chernozhukov, Wuthrich & Zhu (2021) conformal prediction interval for ASCM."""
        lam = self.ridge_lambda
        split = self._split

        def _fit_fn(y_null: np.ndarray, d_mat: np.ndarray) -> np.ndarray:
            # CWZ (2021) requires fitting on the full null-adjusted trajectory for permutation symmetry.
            w_a, _, _, intercept_a = _fit_ascm_weights(y_null, d_mat, ridge_lambda=lam)
            return w_a @ d_mat + intercept_a

        return conformal_sc_interval(
            self.actual_trajectory, self._donors, split, _fit_fn,
            alpha=alpha, point_estimate=self.estimate,
        )

    def summary(self) -> str:
        return (
            "[EXPERIMENTAL] Ridge-Augmented Synthetic Control (Ben-Michael et al. 2021)\n"
            f"Average post-treatment gap: {self.estimate:.5g}\n"
            f"Pre-treatment RMSPE: {self.pre_rmspe:.5g} (ridge lambda={self.ridge_lambda:.4g})\n"
            "Primary SE/CI: unavailable (call .conformal_interval() for finite-sample conformal CI)."
        )

    def plot(self, **kwargs: Any):
        from quasi.visualization.synth_plots import render_synth_figure

        return render_synth_figure(
            time_labels=self._time_labels,
            actual_trajectory=self.actual_trajectory,
            synthetic_trajectory=self.synthetic_trajectory,
            intervention_idx=self._split,
            **kwargs,
        )

    def bibtex(self) -> str:
        return """@article{benmichael2021augmented,
  title={The augmented synthetic control method},
  author={Ben-Michael, Eli and Feller, Avi and Rothstein, Jesse},
  journal={Journal of the American Statistical Association},
  volume={116},
  number={536},
  pages={1789--1803},
  year={2021}
}"""

    def methods_text(self) -> str:
        return (
            f"A Ridge-Augmented Synthetic Control (Ben-Michael, Feller & Rothstein 2021) estimated "
            f"an average post-treatment gap of {self.estimate:.4f} (ridge lambda={self.ridge_lambda:.4g}, "
            f"pre-RMSPE={self.pre_rmspe:.4f})."
        )


class AugmentedSCM(BaseQEDModel):
    """Ridge-Augmented Synthetic Control estimator for imperfect pre-treatment fit."""

    def __init__(
        self,
        data: Any,
        outcome: str,
        unit: str,
        time: str,
        treated_unit: str,
        intervention_time: Any,
        *,
        ridge_lambda: Optional[float] = None,
        treatment: Optional[str] = None,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(data=data, outcome=outcome, alpha=alpha)
        self.raw_data = data
        self.unit = unit
        self.time = time
        self.treated_unit = treated_unit
        self.intervention_time = intervention_time
        self.ridge_lambda = ridge_lambda
        self.treatment = treatment

    def fit(self) -> ASCMResult:
        panel = prepare_synthetic_panel(
            self.raw_data,
            outcome=self.outcome,
            unit=self.unit,
            time=self.time,
            treated_unit=self.treated_unit,
            intervention_time=self.intervention_time,
            treatment=self.treatment,
        )
        k = panel.intervention_idx
        w_ascm, w_scm, lam, intercept_a = _fit_ascm_weights(
            panel.treated[:k], panel.donors[:, :k], ridge_lambda=self.ridge_lambda
        )
        synthetic = w_ascm @ panel.donors + intercept_a
        att = float(np.mean((panel.treated - synthetic)[k:]))
        return ASCMResult(
            estimate=att,
            outcome=self.outcome,
            unit_name=self.treated_unit,
            n_obs=panel.n_rows,
            weights_ascm=dict(zip(panel.donor_ids, map(float, w_ascm))),
            weights_scm=dict(zip(panel.donor_ids, map(float, w_scm))),
            ridge_lambda=lam,
            actual=panel.treated,
            synthetic=synthetic,
            time_labels=panel.time_labels,
            split=k,
            treatment_audit=panel.treatment_audit,
            model=self,
            donors=panel.donors,
        )
