"""Experimental observational ATT estimators with separate matching and AIPW contracts.

Both methods require conditional exchangeability and positivity for a causal
interpretation. These assumptions cannot be established from observed balance.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.errors import ConvergenceError, InsufficientSupportError, InvalidDesignError
from quasi.prep.extractor import extract_column_arrays

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(x, w) / w.sum())


def _balance(x: np.ndarray, d: np.ndarray, covariates: list[str],
             weights: np.ndarray | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """Absolute SMDs on one fixed unadjusted pooled-standard-deviation scale."""
    treated = d == 1
    control = ~treated
    weights = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float)
    before, after = {}, {}
    for j, name in enumerate(covariates):
        v = x[:, j]
        scale = float(np.sqrt((v[treated].var(ddof=1) + v[control].var(ddof=1)) / 2))
        if scale < 1e-12:
            # Identical constants are balanced, not infinitely imbalanced.
            before[name] = 0.0 if abs(v[treated].mean() - v[control].mean()) < 1e-12 else float('inf')
            after[name] = 0.0 if abs(v[treated].mean() - _weighted_mean(v[control], weights[control])) < 1e-12 else float('inf')
        else:
            before[name] = float(abs(v[treated].mean() - v[control].mean()) / scale)
            after[name] = float(abs(v[treated].mean() - _weighted_mean(v[control], weights[control])) / scale)
    return before, after


def _overlap_summary(e: np.ndarray, d: np.ndarray) -> dict[str, float]:
    """Describe empirical score support, without selecting or trimming a population.

    The estimated logit score is a model-based quantity, not evidence that
    unconfoundedness holds. Range intersections are descriptive only.
    """
    treated = e[d == 1]
    control = e[d == 0]
    lo_c, hi_c = float(control.min()), float(control.max())
    lo_t, hi_t = float(treated.min()), float(treated.max())
    return {
        "propensity_treated_min": lo_t,
        "propensity_treated_max": hi_t,
        "propensity_control_min": lo_c,
        "propensity_control_max": hi_c,
        "fraction_treated_outside_control_score_range": float(np.mean((treated < lo_c) | (treated > hi_c))),
        "fraction_control_outside_treated_score_range": float(np.mean((control < lo_t) | (control > hi_t))),
        "empirical_overlap_range_width": max(0.0, min(hi_t, hi_c) - max(lo_t, lo_c)),
    }


def _nearest_controls(scores: np.ndarray, treated_indices: np.ndarray,
                      control_indices: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-nearest PS matches, no quadratic distance matrix.

    Sorting by (absolute score distance, original control row) handles all
    ties, including discrete scores; only a single control-distance vector
    is allocated per treated record. Matching is with replacement *across*
    treated units, without duplicate controls *within* one matched set.
    """
    if n_neighbors > len(control_indices):
        raise InsufficientSupportError("n_neighbors exceeds the number of available controls")
    indices = np.empty((len(treated_indices), n_neighbors), dtype=int)
    distances = np.empty((len(treated_indices), n_neighbors), dtype=float)
    reference = scores[control_indices]
    for row, treated_idx in enumerate(treated_indices):
        dist = np.abs(reference - scores[treated_idx])
        # np.argpartition avoids sorting the entire donor pool, but the kth
        # tie group still needs an original-index tie break.
        threshold = float(np.partition(dist, n_neighbors - 1)[n_neighbors - 1])
        eligible = np.flatnonzero(dist <= threshold)
        ordering = np.lexsort((control_indices[eligible], dist[eligible]))
        chosen = eligible[ordering[:n_neighbors]]
        indices[row] = control_indices[chosen]
        distances[row] = dist[chosen]
    return indices, distances


def _propensity(z: np.ndarray, d: np.ndarray, *, return_coef: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Unpenalized logistic MLE; reject failed optimization and separation."""
    if np.linalg.matrix_rank(z) != z.shape[1]:
        raise InvalidDesignError("Propensity design is rank-deficient")

    # Optimize a scaled design and a mean (rather than sum) objective. This
    # avoids spurious line-search failures for large n or mixed units; no ridge
    # penalty, trimming, or altered propensity estimand is introduced.
    center = z[:, 1:].mean(axis=0)
    scale = z[:, 1:].std(axis=0)
    if np.any(scale < 1e-12) or not np.isfinite(scale).all():
        raise InvalidDesignError("Propensity covariates contain constant or degenerate columns")
    zs = np.column_stack([np.ones(len(d)), (z[:, 1:] - center) / scale])

    def fun(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = zs @ beta
        p = expit(eta)
        loss = np.mean(np.logaddexp(0, eta) - d * eta)
        grad = zs.T @ (p - d) / len(d)
        return float(loss), grad

    fit = minimize(fun, np.zeros(zs.shape[1]), method="BFGS", jac=True,
                   options={"maxiter": 1000, "gtol": 1e-9})
    if np.isfinite(fit.x).all() and np.linalg.norm(fun(fit.x)[1], ord=np.inf) >= 1e-8:
        retry = minimize(fun, fit.x, method="L-BFGS-B", jac=True,
                         options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-11})
        if np.isfinite(retry.x).all() and np.linalg.norm(fun(retry.x)[1], ord=np.inf) < np.linalg.norm(fun(fit.x)[1], ord=np.inf):
            fit = retry
    if not np.all(np.isfinite(fit.x)) or np.linalg.norm(fun(fit.x)[1], ord=np.inf) >= 1e-8:
        raise ConvergenceError("Propensity logistic MLE failed; check separation and covariate scaling")
    e = expit(zs @ fit.x)
    if not np.all(np.isfinite(e)) or e.min() < 0.005 or e.max() > 0.995:
        raise InsufficientSupportError("Estimated propensities approach 0 or 1; no silent clipping or trimming")
    information = (z.T * (e * (1 - e))) @ z
    if np.linalg.matrix_rank(information) != z.shape[1] or np.linalg.cond(information) > 1e12:
        raise InsufficientSupportError("Propensity information is nearly singular (overlap/separation)")
    if return_coef:
        coefficients = np.r_[fit.x[0] - np.sum(fit.x[1:] * center / scale), fit.x[1:] / scale]
        return e, coefficients
    return e


class MatchingResult(QEDResult):
    """Explicit observational ATT result with method-specific inference metadata."""

    def __init__(self, *, estimate: float, se: float, p_value: float, ci: tuple[float, float],
                 outcome_name: str, treatment_name: str, n_obs: int, n_treated: int,
                 smd_before: dict[str, float], smd_after: dict[str, float],
                 diagnostics: dict[str, DiagnosticTest], underlying_model: BaseQEDModel,
                 method: str, extra: dict[str, Any] | None = None) -> None:
        super().__init__(estimate=estimate, se=se, p_value=p_value, ci=ci,
                         estimand="Average Treatment Effect on the Treated (ATT)",
                         outcome_name=outcome_name, model_name=method, n_obs=n_obs,
                         n_treated=n_treated, diagnostics=diagnostics,
                         sub_effects={"treatment_variable": treatment_name,
                                      "smd_before": smd_before, "smd_after": smd_after},
                         model_meta={"balance_diagnostic": "absolute_SMD_fixed_original_pooled_SD", **(extra or {})},
                         underlying_model=underlying_model)
        self.treatment_name = treatment_name
        self.smd_before = smd_before
        self.smd_after = smd_after
        self.inference_details = ({"method": "not_available", "reference_certified": False}
                                  if not np.isfinite(se) else
                                  {"method": ("cross_fitted_AIPW_ATT_EIF_HC0_experimental" if (extra or {}).get("cross_fitted") else
                                              "stacked_logit_IPW_Hajek_HC0" if method == "IPWATT" else
                                              "stacked_logit_control_OLS_ATT_M_estimator_HC0"),
                                   "reference_certified": False})

    def to_metadata_dict(self) -> dict[str, Any]:
        rec = super().to_metadata_dict()
        if not np.isfinite(self.se):
            rec["primary_metric"].update({"se": None, "p_value": None, "ci": (None, None)})
            rec["primary_metric"].pop("ci_95", None)
        return rec

    def summary(self) -> str:
        uncertainty = (f"SE={self.se:.5g}; CI=[{self.ci_lower:.5g}, {self.ci_upper:.5g}]"
                       if np.isfinite(self.se) else "Sampling SE/CI unavailable; point estimate only")
        return (f"[EXPERIMENTAL] {self.model_name} — {self.estimand}\n"
                f"ATT={self.estimate:.6g}; {uncertainty}\n"
                f"Treated={self.n_treated}, total={self.n_obs}\n"
                "Requires conditional exchangeability, overlap and consistency; balance does not prove identification.")

    def ovb_sensitivity(self, *, q: float = 1.0, alpha: float | None = None):
        """Cinelli & Hazlett (2020) omitted-variable bias sensitivity analysis."""
        from quasi.observational.sensitivity import ovb_sensitivity

        if not np.isfinite(self.se):
            raise InvalidDesignError("OVB sensitivity requires a finite SE (use AIPWATT/IPWATT or abadie_imbens_se)")
        k = len(self._model.covariates) + 2 if self._model is not None else 2
        df = max(2, self.n_obs - k)
        return ovb_sensitivity(
            self.estimate, self.se, df,
            q=q, alpha=(1.0 - self.confidence_level) if alpha is None else alpha,
        )

    def abadie_imbens_se(self, *, l_matches: int = 2) -> dict[str, Any]:
        """Abadie & Imbens (2006, Econometrica) Theorem 7 analytical conditional SE for PSM ATT.

        Computes the exact reuse-adjusted matching variance using within-control
        L-nearest-neighbor conditional variance estimation on propensity scores:
            Var_AI = (1 / N_1^2) * [ sum_{i in T} (tau_i - tau)^2
                                     + sum_{j in C} (K_M(j)^2 - K_M(j)/M) * sigma^2_0(e_j) ]
        Note: conditional on estimated propensity scores (Abadie-Imbens 2006); does
        not include first-stage propensity-score estimation adjustment (Abadie-Imbens 2016).
        """
        if self.model_name != "PropensityScoreMatching" or self._model is None:
            raise InvalidDesignError("abadie_imbens_se() applies only to fitted PropensityScoreMatching results")
        if type(l_matches) is not int or l_matches < 1:
            raise InvalidDesignError("l_matches must be a positive integer")
        y, d, _, _, e = self._model._prepare()
        ti = self.matched_treated_indices
        matches = self.matched_control_indices
        if matches.ndim == 1:
            matches = matches[:, None]
        m_nbrs = matches.shape[1]
        n1 = len(ti)
        ci = np.flatnonzero(d == 0)
        if len(ci) <= l_matches:
            raise InsufficientSupportError("Insufficient controls for within-arm conditional variance matching")

        tau_i = y[ti] - y[matches].mean(axis=1)
        term_treated = float(np.sum((tau_i - self.estimate) ** 2))

        # Within-control L-nearest-neighbor variance estimator sigma^2_0(e_j)
        e_c = e[ci]
        y_c = y[ci]
        order = np.argsort(e_c, kind="mergesort")
        e_s = e_c[order]
        y_s = y_c[order]
        n0 = len(ci)
        sig2_c = np.empty(n0, dtype=float)
        for pos in range(n0):
            lo = max(0, pos - l_matches - 2)
            hi = min(n0, pos + l_matches + 3)
            cands = [idx for idx in range(lo, hi) if idx != pos]
            cands.sort(key=lambda idx: abs(e_s[idx] - e_s[pos]))
            nbrs = cands[:l_matches]
            l_eff = len(nbrs)
            sig2_c[order[pos]] = (l_eff / (l_eff + 1.0)) * ((y_s[pos] - float(np.mean(y_s[nbrs]))) ** 2)

        k_weights = self.matched_control_weights[ci]
        # Reuse adjustment: K_M(j)^2 - K_M(j)/M >= 0 identically
        reuse_factor = np.maximum(0.0, k_weights * k_weights - k_weights / m_nbrs)
        term_reuse = float(np.sum(reuse_factor * sig2_c))
        var_ai = (term_treated + term_reuse) / (n1 * n1)
        se_ai = float(np.sqrt(max(var_ai, 0.0)))
        alpha = 1.0 - self.confidence_level
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        p_val = float(2.0 * stats.norm.sf(abs(self.estimate / se_ai))) if se_ai > 0 else float("nan")
        return {
            "estimate": self.estimate,
            "se": se_ai,
            "p_value": p_val,
            "ci_lower": self.estimate - crit * se_ai,
            "ci_upper": self.estimate + crit * se_ai,
            "confidence_level": self.confidence_level,
            "l_matches": l_matches,
            "method": "abadie_imbens_2006_conditional_matching_se",
            "propensity_estimation_adjusted": False,
        }

    def plot(self, **kwargs: Any) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        from quasi.visualization.love_plot import render_love_figure
        return render_love_figure(smd_unadjusted=self.smd_before,
                                  smd_adjusted=self.smd_after, **kwargs)

    def bibtex(self) -> str:
        return """@article{rosenbaum1983central,
 title={The central role of the propensity score in observational studies for causal effects},
 author={Rosenbaum, Paul R and Rubin, Donald B},
 journal={Biometrika}, volume={70}, number={1}, pages={41--55}, year={1983}
}"""

    def methods_text(self) -> str:
        return (f"The experimental {self.model_name} estimator of the ATT for {self.outcome_name} "
                f"was computed with pretreatment covariates. It requires exchangeability and positivity "
                f"conditional on those covariates; neither is established by these data. "
                f"Statistical inference: {self.inference_details['method']} (not independently certified).")


class _ObservationalBase(BaseQEDModel):
    def __init__(self, data: pd.DataFrame | Any, outcome: str, treatment: str,
                 covariates: list[str], alpha: float = 0.05) -> None:
        if not covariates:
            raise InvalidDesignError("At least one pretreatment covariate is required")
        if not 0 < alpha < 1:
            raise InvalidDesignError("alpha must be in (0,1)")
        super().__init__(data=data, outcome=outcome, covariates=covariates, alpha=alpha)
        self.treatment_col = treatment
        self.arrays = extract_column_arrays(data, {
            "outcome": outcome, "treatment": treatment, "covariates": covariates})

    def _prepare_arrays_only(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(self.arrays["outcome"], dtype=float).reshape(-1)
        d = np.asarray(self.arrays["treatment"], dtype=float).reshape(-1)
        x = np.asarray(self.arrays["covariates"], dtype=float)
        n = len(y)
        if x.ndim != 2 or x.shape[0] != n or len(d) != n or not all(np.isfinite(a).all() for a in (y, d, x)):
            raise InvalidDesignError("Observational outcome, treatment and covariates must be complete and finite")
        if not np.isin(d, [0, 1]).all():
            raise InvalidDesignError("Treatment indicator must be binary (0 or 1)")
        z = np.column_stack([np.ones(n), x])
        k = z.shape[1]
        if min(np.sum(d == 1), np.sum(d == 0)) < max(4, k + 2):
            raise InsufficientSupportError("Insufficient treated or control observations for nuisance estimation")
        return y, d, x, z

    def _prepare(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y, d, x, z = self._prepare_arrays_only()
        return y, d, x, z, _propensity(z, d)

    def _result(self, y: np.ndarray, d: np.ndarray, x: np.ndarray, before: dict[str, float],
                after: dict[str, float], estimate: float, se: float, method: str,
                extra: dict[str, Any]) -> MatchingResult:
        if not np.isfinite(estimate):
            raise InvalidDesignError("ATT estimate is not finite")
        if np.isfinite(se):
            if se <= 0:
                raise InsufficientSupportError("Estimated ATT uncertainty is degenerate")
            z = stats.norm.ppf(1 - self.alpha / 2)
            ci = (estimate - z * se, estimate + z * se)
            p = float(2 * stats.norm.sf(abs(estimate / se)))
        else:
            ci = (float("nan"), float("nan"))
            p = float("nan")
        worst = max(after.values())
        status = DiagnosticStatus.WARN if worst > 0.1 else DiagnosticStatus.PASS
        overlap = extra.get("fraction_treated_outside_control_score_range", 0.0)
        overlap_status = DiagnosticStatus.WARN if overlap > 0 else DiagnosticStatus.PASS
        diagnostics = {"Empirical propensity support": DiagnosticTest(
            name="Fraction of treated scores outside observed control-score range",
            status=overlap_status, statistic_name="Outside fraction",
            statistic_value=float(overlap), p_value=None,
            threshold="0 (descriptive only)",
            message="Empirical range overlap does not establish positivity or unconfoundedness",
            remedy="Inspect propensity distributions, covariate overlap, and treatment assignment"),
            "Balance after adjustment": DiagnosticTest(
            name="Absolute standardized mean differences", status=status,
            statistic_name="Maximum adjusted SMD", statistic_value=float(worst),
            p_value=None, threshold="SMD <= 0.1 (descriptive)",
            message="Covariate balance is descriptive and cannot establish exchangeability",
            remedy="Inspect overlap, covariate specification and unobserved confounding")}
        return MatchingResult(estimate=float(estimate), se=float(se), p_value=p,
                              ci=ci, outcome_name=self.outcome,
                              treatment_name=self.treatment_col, n_obs=len(y),
                              n_treated=int(d.sum()), smd_before=before,
                              smd_after=after, diagnostics=diagnostics,
                              method=method, extra=extra, underlying_model=self)



def _stratified_folds(d: np.ndarray, *, n_splits: int, seed: int, min_training: int) -> np.ndarray:
    """Deterministic disjoint held-out folds, stratified independently by treatment.

    Row order is meaningful, so the caller must preserve identical row order to
    reproduce assignments. Every observation is held out exactly once.
    """
    rng = np.random.default_rng(seed)
    assignment = np.full(len(d), -1, dtype=np.int32)
    for group in (0, 1):
        indices = np.flatnonzero(d == group)
        if len(indices) < n_splits:
            raise InsufficientSupportError("Every cross-fit validation fold requires both treatment groups")
        if len(indices) - int(np.ceil(len(indices) / n_splits)) < min_training:
            raise InsufficientSupportError(
                "Cross-fitting would leave insufficient nuisance-training observations in a treatment group")
        for fold, subset in enumerate(np.array_split(rng.permutation(indices), n_splits)):
            assignment[subset] = fold
    if np.any(assignment < 0):
        raise RuntimeError("Cross-fit fold assignment left unassigned observations")
    return assignment


def _crossfit_predictions(y: np.ndarray, d: np.ndarray, z: np.ndarray,
                          *, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train nuisance models on complementary rows only; predict held-out rows.

    The binary logit and control OLS use no information from the validation fold.
    This is not a general machine-learning learner plug-in or a causal certificate.
    """
    n, k = z.shape
    fold_id = _stratified_folds(d, n_splits=n_splits, seed=seed, min_training=max(4, k + 2))
    e = np.full(n, np.nan)
    m0 = np.full(n, np.nan)
    fold_info = []
    for fold in range(n_splits):
        held_out = fold_id == fold
        train = ~held_out
        training_d = d[train]
        z_train = z[train]
        if np.linalg.matrix_rank(z_train[training_d == 0]) < k:
            raise InvalidDesignError(f"Cross-fit fold {fold}: control-outcome training matrix is rank-deficient")
        e[held_out] = expit(_fit_logit_coefficients(z_train, training_d) @ z[held_out].T)
        beta, _, rank, _ = np.linalg.lstsq(z_train[training_d == 0], y[train][training_d == 0], rcond=None)
        if rank != k:
            raise InvalidDesignError(f"Cross-fit fold {fold}: control-outcome regression is rank-deficient")
        m0[held_out] = z[held_out] @ beta
        fold_info.append({"fold": fold, "heldout_n": int(held_out.sum()),
                          "heldout_treated_n": int(d[held_out].sum()),
                          "train_n": int(train.sum()), "train_controls_n": int((training_d == 0).sum()),
                          "train_treated_n": int(training_d.sum())})
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(m0)):
        raise ConvergenceError("Nonfinite out-of-fold nuisance predictions")
    if e.min() < 0.005 or e.max() > 0.995:
        raise InsufficientSupportError("Out-of-fold propensity scores approach 0 or 1; no silent clipping")
    info = {"cross_fitted": True, "n_splits": n_splits, "fold_seed": seed,
            "split_algorithm": "stratified_numpy_PCG64_permutation_array_split_v1",
            "fold_assignment_sha256": hashlib.sha256(fold_id.astype('<i4').tobytes()).hexdigest(),
            "folds": fold_info,
            "nuisance_predictions": "out_of_fold_only",
            "cross_fit_row_order_sensitive": True}
    return e, m0, fold_id, info


def _fit_logit_coefficients(z: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Recover original-scale logit coefficients consistently with _propensity.

    Calls the exact same guarded optimizer with an optional coefficient-return
    path, rather than estimating a second, numerically distinct logit model.
    """
    _, coefs = _propensity(z, d, return_coef=True)
    return coefs


def _crossfit_setup(n_splits: int, fold_seed: int) -> tuple[int, int]:
    if type(n_splits) is not int or n_splits < 1 or n_splits > 20:
        raise InvalidDesignError("n_splits must be an integer in [1, 20]")
    if type(fold_seed) is not int or fold_seed < 0 or fold_seed > (2**32 - 1):
        raise InvalidDesignError("fold_seed must be a nonnegative 32-bit integer")
    return n_splits, fold_seed

class AIPWATT(_ObservationalBase):
    """Traditional observational doubly robust ATT with joint nuisance-score sandwich SE.

    Logistic propensity and linear untreated-outcome regression are estimated on
    the full sample / controls respectively. No cross-fitting or ML nuisance
    guarantees are claimed. The sandwich accounts for fitted nuisance scores.
    """

    def __init__(
        self,
        data: pd.DataFrame | Any,
        outcome: str,
        treatment: str,
        covariates: list[str],
        alpha: float = 0.05,
        *,
        n_splits: int = 1,
        fold_seed: int = 42,
        stabilized: bool = False,
        propensity_learner: Any = None,
        outcome_learner: Any = None,
    ) -> None:
        super().__init__(data, outcome, treatment, covariates, alpha)
        self.n_splits, self.fold_seed = _crossfit_setup(n_splits, fold_seed)
        if (propensity_learner is not None or outcome_learner is not None) and self.n_splits < 2:
            raise InvalidDesignError("Pluggable ML nuisance learners require cross-fitting (n_splits >= 2)")
        self.stabilized = bool(stabilized)
        self.propensity_learner = propensity_learner
        self.outcome_learner = outcome_learner

    def fit(self) -> MatchingResult:
        if self.n_splits > 1:
            return self._fit_cross_fitted()
        y, d, x, z, e = self._prepare()
        n, k = z.shape
        controls = d == 0
        zc = z[controls]
        yc = y[controls]
        beta, _, rank, _ = np.linalg.lstsq(zc, yc, rcond=None)
        if rank != k:
            raise InvalidDesignError("Untreated outcome regression is rank deficient")
        residual = y - z @ beta
        odds = e / (1 - e)
        w = (1 - d) * odds
        p_t = float(np.mean(d))
        if self.stabilized:
            w_bar = float(np.mean(w))
            if w_bar <= 0.0:
                raise InsufficientSupportError("Zero control propensity weights")
            m1 = float(np.mean(d * residual) / p_t)
            m0 = float(np.mean(w * residual) / w_bar)
            tau = float(m1 - m0)
            # Exact stacked M-estimation treating BOTH p_t = E[D] and w_bar = E[w] as estimated:
            # 0: d * (residual - m1)          -> m1 = E[d * r] / E[d]
            # 1: w * (residual - m0)          -> m0 = E[w * r] / E[w]
            # 2..: logit score (d - e) * z, control OLS score (1 - d) * r * z
            moments = np.column_stack([
                d * (residual - m1),
                w * (residual - m0),
                (d - e)[:, None] * z,
                ((1 - d) * residual)[:, None] * z,
            ])
            jac = np.zeros((2 + 2 * k, 2 + 2 * k))
            jac[0, 0] = -p_t
            jac[0, 2 + k:] = -np.mean(d[:, None] * z, axis=0)
            jac[1, 1] = -w_bar
            jac[1, 2:2 + k] = np.mean((w * (residual - m0))[:, None] * z, axis=0)
            jac[1, 2 + k:] = -np.mean(w[:, None] * z, axis=0)
            jac[2:2 + k, 2:2 + k] = -(z.T * (e * (1 - e))) @ z / n
            jac[2 + k:, 2 + k:] = -(zc.T @ zc) / n
        else:
            tau = float(np.mean((d - w) * residual) / p_t)
            # Moment equations, in order: ATT, logit score, control OLS score.
            # Full stacked influence function accounts for estimation of both
            # nuisance models, not only the oracle (known-nuisance) AIPW score.
            moments = np.column_stack([
                d * (residual - tau) - w * residual,
                (d - e)[:, None] * z,
                ((1 - d) * residual)[:, None] * z,
            ])
            jac = np.zeros((1 + 2 * k, 1 + 2 * k))
            jac[0, 0] = -p_t
            jac[0, 1:1 + k] = np.mean((-w * residual)[:, None] * z, axis=0)
            jac[0, 1 + k:] = np.mean((w - d)[:, None] * z, axis=0)
            jac[1:1 + k, 1:1 + k] = -(z.T * (e * (1 - e))) @ z / n
            jac[1 + k:, 1 + k:] = -(zc.T @ zc) / n
        try:
            if np.linalg.cond(jac) > 1e12:
                raise np.linalg.LinAlgError("ill-conditioned moments")
            sol_if = -np.linalg.solve(jac, moments.T)
            if_tau = (sol_if[0] - sol_if[1]) if self.stabilized else sol_if[0]
        except np.linalg.LinAlgError as err:
            raise InsufficientSupportError("Singular ATT/nuisance estimating equations") from err
        se = float(np.sqrt(np.mean((if_tau - if_tau.mean())**2) / n))
        before, after = _balance(x, d, self.covariates, weights=np.where(d == 0, odds, 1.0))
        return self._result(y, d, x, before, after, tau, se, "AIPWATT",
                            {**_overlap_summary(e, d), "propensity_min": float(e.min()), "propensity_max": float(e.max()),
                             "stabilized_hajek_weights": self.stabilized,
                             "effective_control_weight_n": float(w.sum()**2 / np.dot(w, w)),
                             "uncertainty": "stacked_m_estimation_HC0_experimental",
                             "propensity_clipped": False})


    def _fit_cross_fitted(self) -> MatchingResult:
        y, d, x, z = self._prepare_arrays_only()
        if self.propensity_learner is not None or self.outcome_learner is not None:
            from copy import deepcopy

            n, k = z.shape
            fold_id = _stratified_folds(d, n_splits=self.n_splits, seed=self.fold_seed, min_training=max(4, k + 2))
            e = np.full(n, np.nan)
            m0 = np.full(n, np.nan)
            for fold in range(self.n_splits):
                held = fold_id == fold
                train = ~held
                if self.propensity_learner is not None:
                    pl = deepcopy(self.propensity_learner)
                    pl.fit(x[train], d[train].astype(int))
                    probs = np.asarray(pl.predict_proba(x[held]), dtype=float)
                    e[held] = probs[:, 1] if probs.ndim == 2 else probs.reshape(-1)
                else:
                    e[held] = expit(_fit_logit_coefficients(z[train], d[train]) @ z[held].T)
                if self.outcome_learner is not None:
                    ol = deepcopy(self.outcome_learner)
                    ctrl_tr = train & (d == 0)
                    ol.fit(x[ctrl_tr], y[ctrl_tr])
                    m0[held] = np.asarray(ol.predict(x[held]), dtype=float).reshape(-1)
                else:
                    beta, _, rank, _ = np.linalg.lstsq(z[train][d[train] == 0], y[train][d[train] == 0], rcond=None)
                    if rank != k:
                        raise InvalidDesignError(f"Cross-fit fold {fold}: control-outcome regression is rank-deficient")
                    m0[held] = z[held] @ beta
            if not np.all(np.isfinite(e)) or not np.all(np.isfinite(m0)):
                raise ConvergenceError("Nonfinite out-of-fold ML nuisance predictions")
            if e.min() < 0.005 or e.max() > 0.995:
                raise InsufficientSupportError("Out-of-fold propensity scores approach 0 or 1; no silent clipping")
            info = {
                "cross_fitted": True, "n_splits": self.n_splits, "fold_seed": self.fold_seed,
                "ml_nuisance_learners": True,
                "fold_assignment_sha256": hashlib.sha256(fold_id.astype("<i4").tobytes()).hexdigest(),
            }
        else:
            e, m0, fold_id, info = _crossfit_predictions(
                y, d, z, n_splits=self.n_splits, seed=self.fold_seed
            )
        odds = e / (1 - e)
        weights = (1 - d) * odds
        p_t = float(np.mean(d))
        residuals = y - m0
        if self.stabilized:
            w_bar = float(np.mean(weights))
            w_norm = weights / w_bar
            r_w = float(np.mean(w_norm * residuals))
            tau = float(np.mean((d / p_t) * residuals) - r_w)
            influence = (d / p_t) * (residuals - tau) - w_norm * (residuals - r_w) - r_w * (d / p_t - 1.0)
        else:
            tau = float(np.mean((d - weights) * residuals) / p_t)
            influence = (d * (residuals - tau) - weights * residuals) / p_t
        se = float(np.sqrt(np.mean((influence - influence.mean())**2) / len(d)))
        before, after = _balance(x, d, self.covariates, weights=np.where(d == 0, odds, 1.0))
        result = self._result(y, d, x, before, after, tau, se, "AIPWATT",
            {**_overlap_summary(e, d), **info,
             "stabilized_hajek_weights": self.stabilized,
             "propensity_min": float(e.min()), "propensity_max": float(e.max()),
             "effective_control_weight_n": float(weights.sum()**2 / np.dot(weights, weights)),
             "uncertainty": "cross_fitted_orthogonal_score_HC0_experimental",
             "propensity_clipped": False, "sampling_inference_certified": False})
        result.fold_assignments = fold_id.copy()
        result.out_of_fold_propensity = e.copy()
        result.out_of_fold_control_outcome = m0.copy()
        return result


class IPWATT(_ObservationalBase):
    """Experimental Hájek ATT via propensity odds weighting (not doubly robust)."""

    def __init__(self, data: pd.DataFrame | Any, outcome: str, treatment: str,
                 covariates: list[str], alpha: float = 0.05, *,
                 n_splits: int = 1, fold_seed: int = 42) -> None:
        super().__init__(data, outcome, treatment, covariates, alpha)
        self.n_splits, self.fold_seed = _crossfit_setup(n_splits, fold_seed)

    def fit(self) -> MatchingResult:
        if self.n_splits > 1:
            y, d, x, z = self._prepare_arrays_only()
            e, m0, fold_id, info = _crossfit_predictions(y, d, z,
                n_splits=self.n_splits, seed=self.fold_seed)
            w = (1-d) * e/(1-e)
            tau = float(np.mean(y[d == 1]) - np.dot(w, y)/w.sum())
            before, after = _balance(x, d, self.covariates, weights=np.where(d == 0, e/(1-e), 1.0))
            result = self._result(y, d, x, before, after, tau, float('nan'), "IPWATT",
                {**_overlap_summary(e, d), **info,
                 "effective_control_weight_n": float(w.sum()**2/np.dot(w,w)),
                 "propensity_min": float(e.min()), "propensity_max": float(e.max()),
                 "uncertainty": "not_available_crossfit_IPW_nonorthogonal",
                 "propensity_clipped": False, "sampling_inference_certified": False})
            result.fold_assignments = fold_id.copy()
            result.out_of_fold_propensity = e.copy()
            return result
        y, d, x, z, e = self._prepare()
        n, k = z.shape
        w = (1-d)*e/(1-e)
        n1 = d.sum()
        w0 = w.sum()
        m1 = float(np.dot(d, y)/n1)
        m0 = float(np.dot(w, y)/w0)
        tau = m1-m0
        moments = np.column_stack([d*(y-m1), w*(y-m0), (d-e)[:,None]*z])
        jac = np.zeros((k+2,k+2))
        jac[0,0] = -n1/n
        jac[1,1] = -w0/n
        jac[1,2:] = np.mean((w*(y-m0))[:,None]*z, axis=0)
        jac[2:,2:] = -(z.T*(e*(1-e))) @ z/n
        try:
            if np.linalg.cond(jac) > 1e12:
                raise np.linalg.LinAlgError("ill-conditioned IPW estimating equations")
            influences = -np.linalg.solve(jac, moments.T)
        except np.linalg.LinAlgError as err:
            raise InsufficientSupportError("Singular IPW/propensity estimating equations") from err
        if_tau = influences[0] - influences[1]
        se = float(np.sqrt(np.mean((if_tau-if_tau.mean())**2)/n))
        before, after = _balance(x, d, self.covariates,
                                 weights=np.where(d==0, e/(1-e), 1.0))
        return self._result(y,d,x,before,after,tau,se,"IPWATT",
                            {**_overlap_summary(e, d), "propensity_min":float(e.min()), "propensity_max":float(e.max()),
                             "effective_control_weight_n": float(w0*w0/np.dot(w,w)),
                             "uncertainty": "stacked_logit_IPW_Hajek_HC0_experimental",
                             "propensity_clipped":False})


class PropensityScoreMatching(_ObservationalBase):
    """Genuine nearest-neighbor propensity matching ATT; point estimate only.

    Control units may be reused across treated units. A treated unit receives
    the mean outcome of n_neighbors distinct closest controls. Optional caliper
    is an *absolute propensity probability* distance and applies to every pair.
    No treated units are implicitly dropped. Matching SE is unavailable until
    a defensible non-smooth estimator is implemented.
    """

    def __init__(self, data: pd.DataFrame | Any, outcome: str, treatment: str,
                 covariates: list[str], alpha: float = 0.05,
                 caliper: float | None = None, n_neighbors: int = 1) -> None:
        super().__init__(data, outcome, treatment, covariates, alpha)
        if type(n_neighbors) is not int or n_neighbors < 1:
            raise InvalidDesignError("n_neighbors must be a positive integer")
        self.n_neighbors = n_neighbors
        if caliper is not None and (isinstance(caliper, bool) or not isinstance(caliper, (int, float)) or not np.isfinite(caliper) or not 0 < caliper <= 1):
            raise InvalidDesignError("caliper must be an absolute propensity distance in (0,1]")
        self.caliper = caliper

    def fit(self) -> MatchingResult:
        y, d, x, z, e = self._prepare()
        ti = np.flatnonzero(d == 1)
        ci = np.flatnonzero(d == 0)
        matches, distances = _nearest_controls(e, ti, ci, self.n_neighbors)
        if self.caliper is not None and np.any(distances > self.caliper):
            raise InsufficientSupportError(
                "Caliper leaves treated observations without all requested matches; "
                "no silent ATT population change")
        counts = np.bincount(matches.ravel(), minlength=len(y)).astype(float) / self.n_neighbors
        before, after = _balance(x, d, self.covariates, counts)
        tau = float(np.mean(y[ti] - y[matches].mean(axis=1)))
        effective = float(counts.sum()**2 / np.dot(counts, counts))
        result = self._result(y, d, x, before, after, tau, float('nan'),
                              "PropensityScoreMatching",
                              {**_overlap_summary(e, d),
                               "matching": "k_nearest_propensity_with_replacement",
                               "n_neighbors": self.n_neighbors,
                               "caliper_probability_distance": self.caliper,
                               "max_matched_distance": float(distances.max()),
                               "median_matched_distance": float(np.median(distances)),
                               "fraction_exact_score_matches": float(np.mean(distances == 0)),
                               "unique_matched_controls": int(np.count_nonzero(counts)),
                               "effective_matched_control_n": effective,
                               "max_control_reuse_count": int(np.bincount(matches.ravel(), minlength=len(y)).max()),
                               "sampling_inference_certified": False,
                               "naive_bootstrap_valid": False})
        # Keep the original k=1 vector-shaped API; multi-neighbor outputs
        # expose an (n_treated, n_neighbors) matrix with documented ordering.
        result.matched_treated_indices = ti
        result.matched_control_indices = matches[:, 0] if self.n_neighbors == 1 else matches
        result.matched_distances = distances[:, 0] if self.n_neighbors == 1 else distances
        result.matched_control_weights = counts
        return result
