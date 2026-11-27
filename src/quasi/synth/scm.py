"""Experimental classic Synthetic Control (SCM), *not* Synthetic DiD.

Unregularized outcome-trajectory matching on a convex donor simplex. This is a
restricted outcome-only SCM specification; predictor weights / nested V selection,
augmented SCM, and formally calibrated design-based inference are not provided.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from quasi.base import BaseQEDModel, QEDResult, DiagnosticStatus, DiagnosticTest
from quasi.errors import ConvergenceError, InvalidDesignError
from quasi.inference.optimization import solve_simplex_regression
from quasi.synth.panel import prepare_synthetic_panel

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


def _fit_scm_weights(target_pre: np.ndarray, donor_pre: np.ndarray) -> np.ndarray:
    """Optimize pre-treatment MSPE, verifying simplex feasibility and KKT residual."""
    n, t = donor_pre.shape
    if n < 2 or t < 2:
        raise InvalidDesignError("SCM needs at least two controls and two preperiods")
    if not np.isfinite(donor_pre).all() or not np.isfinite(target_pre).all():
        raise InvalidDesignError("SCM requires finite input")
    solution = solve_simplex_regression(donor_pre.T, target_pre, ridge=0.0, intercept=False)
    return solution.weights


class SCMResult(QEDResult):
    """Point-estimate-only result; uncertainty is explicitly unavailable."""
    def __init__(self, *, estimate, outcome, unit_name, n_obs, weights,
                 actual, synthetic, time_labels, split, placebo_ratios,
                 placebo_p_value, model, treatment_audit):
        pre_rmspe = float(np.sqrt(np.mean((actual[:split] - synthetic[:split]) ** 2)))
        post_rmspe = float(np.sqrt(np.mean((actual[split:] - synthetic[split:]) ** 2)))
        fit_scale = float(np.std(actual[:split]))
        hhi = float(sum(w * w for w in weights.values()))
        diagnostic = DiagnosticTest(
            name="Pre-treatment RMSPE (descriptive)", status=DiagnosticStatus.WARN,
            statistic_name="RMSPE", statistic_value=pre_rmspe, p_value=None,
            threshold="No calibrated universal cutoff",
            message="Low pre-treatment prediction error supports fit, not causal identification.")
        super().__init__(estimate=estimate, se=float('nan'), p_value=float('nan'),
                         ci=(float('nan'), float('nan')), estimand="Post-period average SCM gap",
                         outcome_name=outcome, model_name="SyntheticControl", n_obs=n_obs,
                         n_treated=1, diagnostics={"pre_rmspe": diagnostic},
                         sub_effects={"treated_unit": unit_name, "donor_weights": weights,
                             "pre_rmspe": pre_rmspe, "post_rmspe": post_rmspe,
                             "normalized_pre_rmspe": pre_rmspe / fit_scale if fit_scale > 0 else None,
                             "placebo_ratio_tail_fraction": placebo_p_value},
                         model_meta={"n_donors": len(weights), "inference_available": False,
                                     "donor_herfindahl": hhi, "effective_donors": 1. / hhi,
                                     "placebo_count": len(placebo_ratios),
                                     "treatment_audit": treatment_audit}, underlying_model=model)
        self.inference_details = {"method": "not_available", "reference_certified": False,
            "reason": "No calibrated SCM sampling uncertainty; placebo comparisons descriptive"}
        self.treatment_audit = dict(treatment_audit)
        self.donor_weights = weights
        self.actual_trajectory = actual.copy()
        self.synthetic_trajectory = synthetic.copy()
        self.pre_rmspe = pre_rmspe
        self.post_rmspe = post_rmspe
        self.placebo_ratios = np.asarray(placebo_ratios, dtype=float)
        self.placebo_p_value = placebo_p_value
        self._time_labels = list(time_labels)
        self._split = split
        self._model = model

    def leave_one_donor_out(self):
        """Sensitivity to dropping each donor; descriptive, not inferential.

        Refit all remaining donor weights using *only pre-treatment* outcomes.
        Fail clearly if any required counterfactual fit cannot converge.
        """
        import pandas as pd
        panel = prepare_synthetic_panel(self._model.raw_data, outcome=self.outcome_name,
            unit=self._model.unit, time=self._model.time,
            treated_unit=self._model.treated_unit,
            intervention_time=self._model.intervention_time, treatment=self._model.treatment)
        rows = []
        for i, label in enumerate(panel.donor_ids):
            others = np.delete(panel.donors, i, axis=0)
            if len(others) < 2:
                continue
            weights = _fit_scm_weights(panel.treated[:panel.intervention_idx],
                                      others[:, :panel.intervention_idx])
            gap = panel.treated - weights @ others
            rows.append({"excluded_donor": label,
                         "post_average_gap": float(gap[panel.intervention_idx:].mean()),
                         "pre_rmspe": float(np.sqrt(np.mean(gap[:panel.intervention_idx]**2))),
                         "difference_from_full": float(gap[panel.intervention_idx:].mean() - self.estimate)})
        return pd.DataFrame(rows)

    def conformal_interval(self, *, alpha: float = 0.05):
        """Chernozhukov, Wuthrich & Zhu (2021, JASA) moving-block conformal CI."""
        from quasi.synth.conformal import conformal_sc_interval

        panel = prepare_synthetic_panel(
            self._model.raw_data, outcome=self.outcome_name,
            unit=self._model.unit, time=self._model.time,
            treated_unit=self._model.treated_unit,
            intervention_time=self._model.intervention_time, treatment=self._model.treatment,
        )
        split = panel.intervention_idx

        def _fit_fn(y_null: np.ndarray, d_mat: np.ndarray) -> np.ndarray:
            # CWZ (2021) requires fitting on the full null-adjusted series for permutation symmetry.
            w = _fit_scm_weights(y_null, d_mat)
            return w @ d_mat

        return conformal_sc_interval(
            panel.treated, panel.donors, split, _fit_fn,
            alpha=alpha, point_estimate=self.estimate,
        )

    def in_time_placebo(self, *, placebo_idx: Optional[int] = None) -> dict[str, Any]:
        """Backdate intervention inside strictly pre-treatment periods (falsification check)."""
        panel = prepare_synthetic_panel(
            self._model.raw_data, outcome=self.outcome_name,
            unit=self._model.unit, time=self._model.time,
            treated_unit=self._model.treated_unit,
            intervention_time=self._model.intervention_time, treatment=self._model.treatment,
        )
        k_pre = panel.intervention_idx
        p_idx = (k_pre // 2) if placebo_idx is None else placebo_idx
        if p_idx < 2 or p_idx >= k_pre:
            raise InvalidDesignError(f"placebo_idx must be in [2, {k_pre - 1}]")
        w_p = _fit_scm_weights(panel.treated[:p_idx], panel.donors[:, :p_idx])
        gap = panel.treated[:k_pre] - w_p @ panel.donors[:, :k_pre]
        return {
            "placebo_time": panel.time_labels[p_idx],
            "placebo_post_average_gap": float(np.mean(gap[p_idx:])),
            "placebo_pre_rmspe": float(np.sqrt(np.mean(gap[:p_idx] ** 2))),
            "placebo_post_rmspe": float(np.sqrt(np.mean(gap[p_idx:] ** 2))),
        }

    def to_metadata_dict(self) -> dict:
        """Present absent inference as null, not NaN, in lightweight metadata."""
        record = super().to_metadata_dict()
        record["primary_metric"].update({"se": None, "p_value": None,
            "ci": (None, None)})
        record["primary_metric"].pop("ci_95", None)
        return record

    def summary(self) -> str:
        return ("[EXPERIMENTAL] Classic outcome-only Synthetic Control\n"
                f"Average post-treatment gap: {self.estimate:.5g}\n"
                f"Pre-treatment RMSPE: {self.pre_rmspe:.5g}\n"
                "SE / CI / calibrated p-value: NOT AVAILABLE\n"
                f"Exploratory placebo-ratio tail fraction: {self.placebo_p_value}\n"
                "No spillovers, no anticipation and credible donors must be defended independently.")

    def plot(self, **kwargs: Any):
        from quasi.visualization.synth_plots import render_synth_figure
        return render_synth_figure(time_labels=self._time_labels,
            actual_trajectory=self.actual_trajectory,
            synthetic_trajectory=self.synthetic_trajectory,
            intervention_idx=self._split, **kwargs)

    def bibtex(self) -> str:
        return ("@article{abadie2010synthetic, title={Synthetic Control Methods for Comparative Case Studies}, "
                "author={Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens}, "
                "journal={Journal of the American Statistical Association}, year={2010}}")

    def methods_text(self) -> str:
        return ("Experimental outcome-only convex Synthetic Control; nonnegative donor weights "
                "sum to one and minimize mean squared pre-treatment outcome errors. "
                "The reported post-treatment average treated-minus-synthetic gap is descriptive "
                "unless identifying assumptions hold. Standard errors, confidence intervals, "
                "and calibrated p-values were not computed. Donor-placebo ratios are exploratory.")


class SyntheticControl(BaseQEDModel):
    """Classic outcome-only SCM for one treated unit with a common treatment onset.

    No donor may be treated during the analysis window. The estimator cannot
    independently verify this assumption; users must vet the donor pool.
    """
    def __init__(self, data: Any, outcome: str, unit: str, time: str,
                 treated_unit: str, intervention_time: Any,
                 covariates: Optional[list[str]] = None, alpha: float = .05,
                 n_placebos: int = 50, seed: int = 42,
                 treatment: Optional[str] = None):
        if covariates:
            raise NotImplementedError("SCM predictor covariates are not implemented; outcome trajectory only")
        if isinstance(n_placebos, bool) or type(n_placebos) is not int or n_placebos < 0:
            raise InvalidDesignError("n_placebos must be nonnegative integer")
        if isinstance(seed, bool) or type(seed) is not int or seed < 0:
            raise InvalidDesignError("seed must be nonnegative integer")
        super().__init__(data=data, outcome=outcome, alpha=alpha)
        self.raw_data = data
        self.unit = unit
        self.time = time
        self.treated_unit = treated_unit
        self.intervention_time = intervention_time
        self.n_placebos = n_placebos
        self.seed = seed
        self.treatment = treatment

    def fit(self) -> SCMResult:
        panel = prepare_synthetic_panel(self.raw_data, outcome=self.outcome,
            unit=self.unit, time=self.time, treated_unit=self.treated_unit,
            intervention_time=self.intervention_time, treatment=self.treatment)
        k = panel.intervention_idx
        actual = panel.treated
        controls = panel.donors
        weights = _fit_scm_weights(actual[:k], controls[:, :k])
        synthetic = weights @ controls
        gap = actual - synthetic
        att = float(np.mean(gap[k:]))
        treated_pre = float(np.sqrt(np.mean(gap[:k] ** 2)))
        treated_post = float(np.sqrt(np.mean(gap[k:] ** 2)))
        if treated_pre < 1e-10:
            # Ratio undefined for exactly matched pre-periods. Avoid infinite placebo scores.
            placebo_p = None
            ratios = []
        else:
            treated_ratio = treated_post / treated_pre
            ratios = []
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(panel.donor_ids), size=min(self.n_placebos, len(panel.donor_ids)), replace=False)
            for ix in indices:
                target = controls[ix]
                others = np.delete(controls, ix, axis=0)
                if len(others) < 2:
                    continue
                try:
                    p_w = _fit_scm_weights(target[:k], others[:, :k])
                except ConvergenceError:
                    # Failure is NOT a zero-valued placebo. Track used placebo count.
                    continue
                p_gap = target - p_w @ others
                p_pre = float(np.sqrt(np.mean(p_gap[:k] ** 2)))
                if p_pre < 1e-10:
                    continue
                ratios.append(float(np.sqrt(np.mean(p_gap[k:] ** 2)) / p_pre))
            placebo_p = float((1 + np.sum(np.array(ratios) >= treated_ratio)) / (1 + len(ratios))) if ratios else None
        return SCMResult(estimate=att, outcome=self.outcome, unit_name=self.treated_unit,
            n_obs=panel.n_rows, weights=dict(zip(panel.donor_ids, map(float, weights))),
            actual=actual, synthetic=synthetic, time_labels=panel.time_labels,
            split=k, placebo_ratios=ratios, placebo_p_value=placebo_p, model=self,
            treatment_audit=panel.treatment_audit)
