"""Experimental, single-treated-unit Synthetic Difference-in-Differences.

The point estimator uses the centered, regularized unit/time simplex objectives of
Arkhangelsky et al. (2021), as exposed by R synthdid's ``synthdid_estimate``
(no time-varying covariates and no sparsification). Cross-language equivalence
and sampling inference have *not* been independently certified.

Unlike previous versions, this module does not present leave-donor-out spread as
an inferential jackknife SE or turn donor placebos into a calibrated p-value.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.errors import ConvergenceError, InvalidDesignError
from quasi.inference.optimization import solve_simplex_regression
from quasi.synth.panel import prepare_synthetic_panel


def _simplex_regression(A: np.ndarray, b: np.ndarray, ridge: float, *,
                        intercept: bool = True) -> tuple[np.ndarray, float]:
    """Minimize (1/m)||b-a-Aw||^2 + ridge**2||w||^2 on simplex via shared solver."""
    sol = solve_simplex_regression(A, b, ridge=ridge, intercept=intercept)
    return sol.weights, sol.intercept


def _sparsify_weights(w: np.ndarray) -> np.ndarray:
    """R synthdid default sparsification: zero weights <= max(w)/4 and renormalize."""
    thresh = float(np.max(w)) / 4.0
    sparse = np.where(w <= thresh, 0.0, w)
    total = float(sparse.sum())
    return (sparse / total) if total > 0 else w


def _sdid_weights(treated: np.ndarray, donors: np.ndarray, split: int, *,
                  zeta_omega: Optional[float] = None,
                  zeta_lambda: Optional[float] = None,
                  sparsify: str = "none") -> tuple[np.ndarray, np.ndarray, dict]:
    """Reference-aligned SDiD simplex objectives (single treated unit).

    R synthdid defaults: noise sd from first differences among donors before
    treatment, eta_omega=(N_treated*T_post)**(1/4), eta_lambda=1e-6. The
    intercept is unpenalized in *both* simplex problems. ``sparsify='default'``
    applies R synthdid's two-step weight thresholding (w <= max(w)/4).
    """
    if sparsify not in ("none", "default"):
        raise InvalidDesignError("sparsify must be 'none' or 'default'")
    treated = np.asarray(treated, dtype=float)
    donors = np.asarray(donors, dtype=float)
    if donors.ndim != 2 or treated.shape != (donors.shape[1],) or split < 2 or split >= len(treated):
        raise InvalidDesignError("Invalid SDiD donor panel or pre/post split")
    if donors.shape[0] < 2 or not np.isfinite(donors).all() or not np.isfinite(treated).all():
        raise InvalidDesignError("SDiD needs two clean donors and finite outcomes")
    # R: sd(apply(Y_control_pre, 1, diff)); sample sd pooled across donor/time.
    differences = np.diff(donors[:, :split], axis=1).ravel()
    noise = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    noise = max(noise, 0.0)
    z_omega = float(noise * (len(treated) - split) ** .25 if zeta_omega is None else zeta_omega)
    z_lambda = float(noise * 1e-6 if zeta_lambda is None else zeta_lambda)
    if not np.isfinite(z_omega) or z_omega < 0 or not np.isfinite(z_lambda) or z_lambda < 0:
        raise InvalidDesignError("SDiD regularization scales must be finite and nonnegative")
    # Target = treated pretreatment trajectory; features = donor trajectories.
    omega, omega_intercept = _simplex_regression(
        donors[:, :split].T, treated[:split], z_omega, intercept=True)
    # Target = average donor posttreatment outcome; features = donor pre-periods.
    lam, lambda_intercept = _simplex_regression(
        donors[:, :split], donors[:, split:].mean(axis=1), z_lambda, intercept=True)
    if sparsify == "default":
        omega = _sparsify_weights(omega)
        lam = _sparsify_weights(lam)
        omega_intercept = float(np.mean(treated[:split]) - np.mean((omega @ donors[:, :split])))
    return omega, lam, {
        "noise_sd_first_differences": noise, "zeta_omega": z_omega,
        "zeta_lambda": z_lambda, "omega_intercept": omega_intercept,
        "lambda_intercept": lambda_intercept,
        "weight_objective": "intercept_centered_ridge_simplex",
        "sparsification": sparsify,
    }


def _sdid_effect(treated: np.ndarray, donors: np.ndarray, split: int,
                 omega: np.ndarray, lam: np.ndarray) -> float:
    """Weighted post minus weighted pre, treated minus synthetic donor mean."""
    g = treated - omega @ donors
    return float(np.mean(g[split:]) - lam @ g[:split])


class SDiDResult(QEDResult):
    """Experimental SDiD point effect without justified sampling uncertainty."""

    def __init__(self, *, estimate, outcome, treated_unit, n_obs,
                 donor_weights, time_weights, weight_details, actual, synthetic,
                 time_labels, split, pre_rmspe, raw_pre_rmspe, placebo_gaps, placebo_effects,
                 placebo_tail_fraction, attempted_placebos, jackknife_spread,
                 treatment_audit, model):
        diagnostic = DiagnosticTest(
            name="Intercept-adjusted pre-treatment fit (descriptive)",
            status=DiagnosticStatus.WARN, statistic_name="RMSPE",
            statistic_value=pre_rmspe, p_value=None,
            threshold="No calibrated scale-independent threshold",
            message="Fit alone does not establish common trends or donor eligibility")
        donor_hhi = float(sum(w*w for w in donor_weights.values()))
        super().__init__(estimate=estimate, se=float("nan"), p_value=float("nan"),
                         ci=(float("nan"), float("nan")),
                         estimand="Single-unit post-period average SDiD effect",
                         outcome_name=outcome, model_name="SyntheticDiD", n_obs=n_obs,
                         n_treated=1, diagnostics={"pre_fit": diagnostic},
                         sub_effects={"treated_unit": treated_unit,
                                      "donor_weights": donor_weights,
                                      "time_weights": time_weights,
                                      "pre_rmspe_intercept_adjusted": pre_rmspe,
                                      "pre_rmspe_raw_levels": raw_pre_rmspe,
                                      "placebo_tail_fraction": placebo_tail_fraction,
                                      "leave_donor_out_spread": jackknife_spread},
                         model_meta={"n_donors": len(donor_weights),
                                     "n_pre_periods": split,
                                     "n_post_periods": len(time_labels) - split,
                                     "donor_effective_n": 1/donor_hhi,
                                     "placebos_attempted": attempted_placebos,
                                     "placebos_succeeded": len(placebo_effects),
                                     "inference_available": False,
                                     "treatment_audit": treatment_audit, **weight_details},
                         underlying_model=model)
        self.inference_details = {
            "method": "not_available", "reference_certified": False,
            "reason": "Single-treated-unit sampling variance and synthdid jackknife parity not validated",
            "placebo_inference": "descriptive_unadjusted_donor_placebos",
        }
        self.treatment_audit = dict(treatment_audit)
        self.donor_weights = donor_weights
        self.time_weights = time_weights
        self.weight_details = weight_details
        self.treated_unit_name = treated_unit
        self.pre_rmspe = pre_rmspe
        self.raw_pre_rmspe = raw_pre_rmspe
        self.placebo_p_val = placebo_tail_fraction  # compatibility: descriptive ONLY
        self.placebo_effects = np.array(placebo_effects, dtype=float)
        self.jackknife_sensitivity_spread = jackknife_spread
        self._actual = actual.copy()
        self._synthetic = synthetic.copy()
        self._time_labels = list(time_labels)
        self._interv_idx = split
        self._placebo_gaps = np.array(placebo_gaps, dtype=float) if placebo_gaps else None

    def to_metadata_dict(self):
        d = super().to_metadata_dict()
        d["primary_metric"].update({"se": None, "p_value": None, "ci": (None, None)})
        d["primary_metric"].pop("ci_95", None)
        return d

    def experimental_placebo_variance(self, *, replications: int = 200,
                                      seed: int = 42) -> dict:
        """Reference-*style* donor placebo spread; NOT a certified sampling SE.

        For a single treated unit, R synthdid's placebo algorithm permutes the
        N0 control rows *without replacement*, reassigns one to pseudo treatment,
        and re-estimates weights. A point estimate with a finite spread is not a
        calibrated CI or identification guarantee; no result inference field
        is updated by this method. No treated outcomes enter any pseudo fit.
        """
        if type(replications) is not int or replications < 20:
            raise InvalidDesignError("Experimental placebo variance requires >=20 integer replications")
        if type(seed) is not int or seed < 0:
            raise InvalidDesignError("Placebo resampling seed must be a nonnegative integer")
        panel = prepare_synthetic_panel(
            self._model.raw_data, outcome=self.outcome_name,
            unit=self._model.unit_col, time=self._model.time_col,
            treated_unit=self._model.treated_unit,
            intervention_time=self._model.intervention_time_raw,
            treatment=self._model.treatment)
        n0, k = panel.donors.shape[0], panel.intervention_idx
        if n0 < 3:
            raise InvalidDesignError("Placebo resampling needs >=3 observed untreated donor units")
        rng = np.random.default_rng(seed)
        estimates = np.empty(replications, dtype=float)
        for r in range(replications):
            sampled = rng.permutation(n0)
            control, pseudo = panel.donors[sampled[:-1]], panel.donors[sampled[-1]]
            try:
                w, lam, _ = _sdid_weights(pseudo, control, k, sparsify=self._model.sparsify)
                estimates[r] = _sdid_effect(pseudo, control, k, w, lam)
            except (ConvergenceError, InvalidDesignError) as exc:
                raise ConvergenceError(
                    f"Placebo replicate {r} did not fit; candidate variance unavailable"
                ) from exc
        if not np.isfinite(estimates).all():
            raise ConvergenceError("Nonfinite donor placebo estimates; no candidate variance available")
        # R synthdid placebo_se: sqrt((B-1)/B) * sd(theta_1,...,theta_B).
        variance = float((replications-1)/replications * np.var(estimates, ddof=1))
        return {
            "status": "experimental_descriptive_only",
            "reference_algorithm": "R_synthdid_placebo_se_structure_no_parity_claim",
            "variance_candidate": variance,
            "spread_candidate": float(np.sqrt(variance)),
            "replications": replications,
            "seed": seed,
            "mean_pseudo_effect": float(estimates.mean()),
            "treatment_history_audit": dict(self.treatment_audit),
            "sampling_inference_certified": False,
            "confidence_interval_available": False,
            "warning": ("Placebo resampling assumes exchangeable untreated donors and is "
                        "not validated for Quasi; not a standard error or CI"),
        }

    def conformal_interval(self, *, alpha: float = 0.05):
        """Chernozhukov, Wuthrich & Zhu (2021, JASA) moving-block conformal CI for SDiD."""
        from quasi.synth.conformal import conformal_sc_interval

        panel = prepare_synthetic_panel(
            self._model.raw_data, outcome=self.outcome_name,
            unit=self._model.unit_col, time=self._model.time_col,
            treated_unit=self._model.treated_unit,
            intervention_time=self._model.intervention_time_raw,
            treatment=self._model.treatment,
        )
        split = panel.intervention_idx
        sp = getattr(self._model, "sparsify", "none")

        zeta_w = float(self.weight_details.get("zeta_omega", 0.0))

        def _fit_fn(y_null: np.ndarray, d_mat: np.ndarray) -> np.ndarray:
            # CWZ (2021) requires fitting on the full null-adjusted series for permutation symmetry.
            sol = solve_simplex_regression(d_mat.T, y_null, ridge=zeta_w, intercept=True)
            w = _sparsify_weights(sol.weights) if sp == "default" else sol.weights
            intercept_w = float(np.mean(y_null - w @ d_mat))
            return w @ d_mat + intercept_w

        return conformal_sc_interval(
            panel.treated, panel.donors, split, _fit_fn,
            alpha=alpha, point_estimate=self.estimate,
        )

    def summary(self) -> str:
        return ("[EXPERIMENTAL] Synthetic Difference-in-Differences\n"
                f"Post-period average effect: {self.estimate:.5g}\n"
                f"Pre-treatment RMSPE: {self.pre_rmspe:.5g}\n"
                f"SE / CI / calibrated p-value: NOT AVAILABLE\n"
                f"Descriptive placebo tail fraction: {self.placebo_p_val}\n"
                "Treatment effects require credible never-treated donors and no spillovers.")

    def plot(self, **kwargs):
        from quasi.visualization.synth_plots import render_synth_figure
        return render_synth_figure(time_labels=self._time_labels,
                                   actual_trajectory=self._actual,
                                   synthetic_trajectory=self._synthetic,
                                   intervention_idx=self._interv_idx,
                                   placebo_gaps=self._placebo_gaps, **kwargs)

    def bibtex(self) -> str:
        return ("@article{arkhangelsky2021synthetic, title={Synthetic Difference-in-Differences}, "
                "author={Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David and Imbens, Guido and Wager, Stefan}, "
                "journal={American Economic Review}, year={2021}}")

    def methods_text(self) -> str:
        sp_mode = self.weight_details.get("sparsification", "none")
        sp_note = (
            "R synthdid-style weight sparsification was applied."
            if sp_mode != "none"
            else "No sparsification was applied."
        )
        return ("Exploratory single-treated-unit Synthetic Difference-in-Differences (Arkhangelsky et al., 2021). "
                "Unit and time simplex weights used an unpenalized intercept and regularization scaled "
                f"by the control first-difference noise estimate. {sp_note} "
                "Point effect contrasts post-period treated-minus-donor outcomes against a weighted "
                "pre-period gap. Statistical uncertainty and calibrated p-values unavailable; "
                "leave-donor-out and donor-placebo calculations are descriptive only. "
                "Reference R synthdid equivalence and uncertainty certification remain outstanding.")


class SyntheticDiD(BaseQEDModel):
    """Single-treated-unit SDiD; all donor units must be untreated throughout.

    Donor eligibility / spillovers cannot be inferred from an outcome-only panel.
    Callers are responsible for establishing the untreated donor pool.
    """
    def __init__(self, data: Any, outcome: str, unit: str, time: str,
                 treated_unit: str, intervention_time: Any,
                 covariates: Optional[list[str]] = None, alpha: float = .05,
                 n_placebos: int = 50, seed: int = 42,
                 treatment: Optional[str] = None,
                 sparsify: str = "none"):
        if covariates:
            raise NotImplementedError("SyntheticDiD covariates do not enter the estimator; covariate adjustment is not implemented")
        if type(n_placebos) is not int or n_placebos < 0:
            raise InvalidDesignError("n_placebos must be a nonnegative integer")
        if type(seed) is not int or seed < 0:
            raise InvalidDesignError("seed must be a nonnegative integer")
        if sparsify not in ("none", "default"):
            raise InvalidDesignError("sparsify must be 'none' or 'default'")
        super().__init__(data=data, outcome=outcome, alpha=alpha)
        self.raw_data = data
        self.unit_col = unit
        self.time_col = time
        self.treated_unit = treated_unit
        self.intervention_time_raw = intervention_time
        self.n_placebos = n_placebos
        self.seed = seed
        self.treatment = treatment
        self.sparsify = sparsify

    def fit(self) -> SDiDResult:
        panel = prepare_synthetic_panel(self.raw_data, outcome=self.outcome,
                                        unit=self.unit_col, time=self.time_col,
                                        treated_unit=self.treated_unit,
                                        intervention_time=self.intervention_time_raw,
                                        treatment=self.treatment)
        k = panel.intervention_idx
        treated, donors = panel.treated, panel.donors
        omega, lam, detail = _sdid_weights(treated, donors, k, sparsify=self.sparsify)
        effect = _sdid_effect(treated, donors, k, omega, lam)
        # The unit-weight objective profiles out a donor-vs-treated level offset.
        # Plot and measure its pre-fit on that fitted scale, without changing ATT.
        synthetic_raw = omega @ donors
        synthetic = synthetic_raw + detail["omega_intercept"]
        raw_pre_rmspe = float(np.sqrt(np.mean((treated[:k] - synthetic_raw[:k])**2)))
        pre_rmspe = float(np.sqrt(np.mean((treated[:k] - synthetic[:k])**2)))
        # Leave-donor-out sensitivity; this is NOT the synthdid jackknife vcov,
        # whose validity for a single treated unit must be assessed separately.
        jack_effects = []
        for j in range(len(panel.donor_ids)):
            if len(panel.donor_ids) <= 2:
                break
            others = np.delete(donors, j, axis=0)
            try:
                wj, lj, _ = _sdid_weights(treated, others, k, sparsify=self.sparsify)
            except ConvergenceError:
                continue
            jack_effects.append(_sdid_effect(treated, others, k, wj, lj))
        jack_spread = (float(np.std(jack_effects, ddof=1))
                       if len(jack_effects) >= 2 else None)

        placebo_gaps, placebo_effects = [], []
        rng = np.random.default_rng(self.seed)
        requested = min(self.n_placebos, len(panel.donor_ids))
        sampled = rng.choice(len(panel.donor_ids), size=requested, replace=False)
        for j in sampled:
            others = np.delete(donors, j, axis=0)
            if len(others) < 2:
                continue
            try:
                wp, lp, p_detail = _sdid_weights(donors[j], others, k, sparsify=self.sparsify)
            except ConvergenceError:
                continue
            placebo_gaps.append(donors[j] - (wp @ others + p_detail["omega_intercept"]))
            placebo_effects.append(_sdid_effect(donors[j], others, k, wp, lp))
        # Descriptive rank fraction, not an exact randomization p-value.
        frac = (float((1 + np.count_nonzero(np.abs(placebo_effects) >= abs(effect))) /
                      (1 + len(placebo_effects))) if placebo_effects else None)
        return SDiDResult(estimate=effect, outcome=self.outcome,
                          treated_unit=self.treated_unit, n_obs=panel.n_rows,
                          donor_weights=dict(zip(panel.donor_ids, map(float, omega))),
                          time_weights=dict(zip(panel.time_labels[:k], map(float, lam))),
                          weight_details=detail, actual=treated, synthetic=synthetic,
                          time_labels=panel.time_labels, split=k,
                          pre_rmspe=pre_rmspe, raw_pre_rmspe=raw_pre_rmspe,
                          placebo_gaps=placebo_gaps,
                          placebo_effects=placebo_effects,
                          placebo_tail_fraction=frac, attempted_placebos=requested,
                          jackknife_spread=jack_spread,
                          treatment_audit=panel.treatment_audit, model=self)


SDiD = SyntheticDiD
