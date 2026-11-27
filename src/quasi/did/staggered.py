"""Experimental balanced-panel group-time DiD (unadjusted/traditional DR).

Supports observed-period anticipation windows, universal/varying preperiod
baselines, and baseline-covariate-adjusted traditional panel DR. The method is
NOT certified equivalent to R did or DRDID; inference is experimental.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.prep.period_index import SmartPeriodIndex
from quasi.did.conditional import improved_dr_panel, traditional_dr_panel
from quasi.did.inference import InferencePlan
from quasi.did.sensitivity import HonestDiDResult, honest_did_sensitivity

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class StaggeredDiDResult(QEDResult):
    """Group-time effects and sample-conditional fixed-composition inference."""

    def __init__(self, *, estimate: float, se: float, p_value: float, ci: Tuple[float, float],
                 outcome_name: str, n_obs: int, n_treated: int,
                 event_study_effects: Dict[int, float], event_study_se: Dict[int, float],
                 simultaneous_ci: Dict[int, Tuple[float, float]],
                 group_time_effects: Dict[Tuple[int, int], float],
                 group_time_se: Dict[Tuple[int, int], float],
                 cell_support: Dict[Tuple[int, int], Dict[str, Any]],
                 excluded_cells: List[Dict[str, Any]],
                 cohort_sizes: Dict[int, int],
                 cohort_effects: Dict[int, float], cohort_se: Dict[int, float],
                 calendar_effects: Dict[int, float], calendar_se: Dict[int, float],
                 control_group: str, anticipation: int, base_period: str, est_method: str,
                 inference_plan: InferencePlan, cluster_col: Optional[str],
                 diagnostics: Dict[str, DiagnosticTest],
                 period_index: SmartPeriodIndex, underlying_model: BaseQEDModel,
                 event_study_cov: Optional[np.ndarray] = None,
                 unconditional_se: Optional[float] = None) -> None:
        super().__init__(
            estimate=estimate, se=se, p_value=p_value, ci=ci,
            estimand="Post-treatment cohort-size-weighted ATT across identified (g,t) cells",
            outcome_name=outcome_name, model_name="CallawaySantAnnaDiD",
            n_obs=n_obs, n_treated=n_treated, diagnostics=diagnostics,
            sub_effects={"event_study_effects": event_study_effects,
                         "event_study_se": event_study_se},
            model_meta={
                "cadence": period_index.cadence, "control_group": control_group,
                "base_period": base_period, "anticipation": anticipation,
                "est_method": est_method,
                "inference_method": ("cluster_level_analytical_if_experimental" if cluster_col is not None else
                    ("improved_dr_ipt_wls_analytical_if_experimental" if est_method == "dr_improved" else
                    ("nuisance_aware_analytical_if_experimental" if est_method == "dr" else
                    "unit_level_analytical_if_fixed_cohort_shares_experimental"))),
                "cluster": cluster_col, "n_clusters": inference_plan.n_clusters if cluster_col else None,
                "cluster_variance_correction": inference_plan.correction if cluster_col else None,
                "simultaneous_band_method": "rademacher_independent_cluster_multiplier_max_t_experimental" if cluster_col else "rademacher_unit_multiplier_max_t_experimental",
                "conditioning": "Observed cohort sizes treated as fixed for aggregation inference",
                "supported_post_cells": sum(t >= g for g, t in group_time_effects),
                "supported_anticipatory_cells": sum(g - anticipation <= t < g for g,t in group_time_effects),
                "skipped_cell_count": len(excluded_cells),
                "method_validation": "experimental_unverified",
                "propensity_overlap_bounds": [0.005, 0.995] if est_method == "dr_improved" else ([0.01, 0.99] if est_method == "dr" else None),
            }, underlying_model=underlying_model,
        )
        self.event_study_effects = event_study_effects
        self.event_study_se = event_study_se
        self.simultaneous_ci = simultaneous_ci
        self.group_time_effects = group_time_effects
        self.group_time_se = group_time_se
        self.cell_support = cell_support
        self.excluded_cells = excluded_cells
        self.cohort_sizes = cohort_sizes
        self.cohort_effects = cohort_effects
        self.cohort_se = cohort_se
        self.calendar_effects = calendar_effects
        self.calendar_se = calendar_se
        self.control_group = control_group
        self.anticipation = anticipation
        self.base_period = base_period
        self.est_method = est_method
        self.cluster_col = cluster_col
        self.period_index = period_index
        self.event_study_cov = event_study_cov
        self.unconditional_se = unconditional_se
        self.inference_details = {
            "method": "cluster_level_conditional_if_analytical_t_experimental" if cluster_col else
                ("improved_dr_ipt_wls_analytical_normal_experimental" if est_method == "dr_improved" else
                ("nuisance_aware_dr_traditional_analytical_normal_experimental" if est_method == "dr" else
                "unit_level_conditional_influence_function_analytical_normal")),
            "reference_certified": False,
            "se_unit": "cluster" if cluster_col else "panel_unit",
            "cluster_field": cluster_col,
            "n_independent_clusters": inference_plan.n_clusters if cluster_col else None,
            "degrees_of_freedom": inference_plan.n_clusters - 1 if cluster_col else None,
            "cluster_variance_correction": inference_plan.correction if cluster_col else None,
            "cohort_weight_uncertainty": "conditioned_on_observed_cohort_sizes",
            "event_band": "rademacher_multiplier_max_t_experimental",
            "control_group": control_group,
            "base_period": base_period, "anticipation": anticipation,
                "est_method": est_method,
        }

    def group_time_frame(self) -> pd.DataFrame:
        """One row per identified (cohort, time) effect, retaining raw period labels."""
        rows = []
        for (g, t), att in sorted(self.group_time_effects.items()):
            rows.append({
                "cohort": self.period_index.idx_to_time[g],
                "time": self.period_index.idx_to_time[t],
                "event_time": t - g,
                "estimate": att,
                "se": self.group_time_se[(g, t)],
                "n_treated_clusters": self.cell_support[(g, t)]["n_treated_clusters"],
                "n_control_clusters": self.cell_support[(g, t)]["n_control_clusters"],
                "n_treated": self.cell_support[(g, t)]["n_treated"],
                "n_controls": self.cell_support[(g, t)]["n_controls"],
                "base_time": self.period_index.idx_to_time[self.cell_support[(g, t)]["base_t"]],
                "is_post": t >= g,
                "is_anticipatory": g - self.anticipation <= t < g,
                "min_fitted_propensity": self.cell_support[(g, t)].get("min_fitted_propensity"),
                "max_fitted_propensity": self.cell_support[(g, t)].get("max_fitted_propensity"),
            })
        return pd.DataFrame(rows)

    def aggregation_frame(self, dimension: Literal["cohort", "calendar"] = "cohort") -> pd.DataFrame:
        """Cohort- or calendar-specific averages of *identified post* cells.

        Cohort aggregates use equal weights over available post-periods;
        calendar aggregates weight contributing cohorts by their observed size.
        The composition and number of cells are explicit to prevent accidental
        comparisons of different exposure horizons.
        """
        if dimension not in ("cohort", "calendar"):
            raise ValueError("dimension must be 'cohort' or 'calendar'")
        values = self.cohort_effects if dimension == "cohort" else self.calendar_effects
        standard_errors = self.cohort_se if dimension == "cohort" else self.calendar_se
        rows = []
        for idx, value in sorted(values.items()):
            cells = [k for k in self.group_time_effects if k[1] >= k[0]
                     and (k[0] if dimension == "cohort" else k[1]) == idx]
            rows.append({dimension: self.period_index.idx_to_time[idx],
                         "estimate": value, "se": standard_errors[idx],
                         "identified_post_cells": len(cells),
                         "contributing_treated_units": sum(self.cohort_sizes[g] for g in {g for g, t in cells})})
        return pd.DataFrame(rows)

    def sensitivity_analysis(
        self,
        *,
        method: Literal["relative_magnitudes", "smoothness"] = "relative_magnitudes",
        m_grid: Optional[List[float]] = None,
        post_weights: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> HonestDiDResult:
        """Rambachan & Roth (2023) HonestDiD pre-trends sensitivity analysis."""
        if self.event_study_cov is None:
            raise ValueError("Full event-study covariance matrix is required for HonestDiD sensitivity")
        ordered = sorted(self.event_study_effects)
        return honest_did_sensitivity(
            event_times=ordered,
            estimates=[self.event_study_effects[e] for e in ordered],
            covariance=self.event_study_cov,
            reference_event=-(self.anticipation + 1),
            base_period=self.base_period,
            method=method,
            m_grid=m_grid,
            post_weights=post_weights,
            alpha=(1.0 - self.confidence_level) if alpha is None else alpha,
        )

    def summary(self) -> str:
        lines = [
            "[EXPERIMENTAL] Group-time DiD; R parity and inference certification pending.",
            "=" * 70,
            f"Staggered DiD | Outcome: {self.outcome_name}",
            "=" * 70,
            f"Identified-cell-weighted ATT: {self.estimate:+.4f} (SE: {self.se:.4f})",
            f"p-value: {self.p_value:.4f}",
            f"{100*self.confidence_level:.0f}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Observations: {self.n_obs:,}; adopting units: {self.n_treated:,}",
            f"Comparison: {self.control_group}; estimator: {self.est_method}; anticipation: {self.anticipation}; skipped cells: {len(self.excluded_cells)}",
            f"Event study (preperiods: {self.base_period} baseline; anticipation window is not a placebo):",
            f"  Event time         ATT       SE       {100*self.confidence_level:.0f}% simultaneous band",
        ]
        for e in sorted(self.event_study_effects):
            low, high = self.simultaneous_ci[e]
            lines.append(f"  {e:+5d}           {self.event_study_effects[e]:+.4f}    "
                         f"{self.event_study_se[e]:.4f}   [{low:+.4f}, {high:+.4f}]")
        lines += ["Interpretation requires untreated parallel trends and no anticipation.", "=" * 70]
        return "\n".join(lines)

    def plot(self, uniform_ci: bool = True, **kwargs: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        from quasi.visualization.event_study import render_event_study_figure
        periods = sorted(self.event_study_effects)
        return render_event_study_figure(
            periods=periods,
            estimates=[self.event_study_effects[p] for p in periods],
            se=[self.event_study_se[p] for p in periods],
            ci_simultaneous=[self.simultaneous_ci[p] for p in periods] if uniform_ci else None,
            alpha=1.0 - self.confidence_level,
            reference_period=-(self.anticipation + 1),
            **kwargs,
        )

    def bibtex(self) -> str:
        from quasi.reporting.provenance import generate_bibtex
        return generate_bibtex(self.model_name)

    def methods_text(self) -> str:
        return (
            "An experimental balanced-panel group-time difference-in-differences "
            "estimator uses the last clean pretreatment period as baseline for post-adoption "
            f"effects and {self.base_period} preperiod comparisons with anticipation={self.anticipation}. "
            f"Controls: {self.control_group}; estimator: {self.est_method}. Aggregation weights cohort size per eligible cell; "
            "unsupported cells are reported and excluded. Panel-unit influence functions "
            "are aggregated by independent cluster if requested, and condition on observed cohort composition; Rademacher multiplier max-t bands "
            "are exploratory. Reference parity and inferential coverage are not certified. "
            "Pretrend tests cannot establish identifying parallel trends."
        )


class StaggeredDiD(BaseQEDModel):
    """Experimental group-time ATT for absorbing-treatment balanced panels."""

    def __init__(self, data: Union[pd.DataFrame, Any], outcome: str, unit: str, time: str,
                 treatment: Optional[str] = None, first_treated: Optional[str] = None,
                 covariates: Optional[List[str]] = None, cluster: Optional[str] = None,
                 control_group: Literal["never_treated", "not_yet_treated"] = "never_treated",
                 b_reps: int = 9999, alpha: float = 0.05, seed: int = 42,
                 anticipation: int = 0, base_period: Literal["varying", "universal"] = "varying",
                 est_method: Literal["unadjusted", "dr", "dr_improved"] = "unadjusted") -> None:
        if est_method not in ("unadjusted", "dr", "dr_improved"):
            raise ValueError("est_method must be unadjusted, dr, or dr_improved")
        if covariates and est_method not in ("dr", "dr_improved"):
            raise NotImplementedError("StaggeredDiD covariates require est_method='dr' or 'dr_improved'; unadjusted does not adjust")
        if est_method in ("dr", "dr_improved") and not covariates:
            raise ValueError("est_method='dr'/'dr_improved' requires nonempty baseline covariates")
        if type(anticipation) is not int or anticipation < 0:
            raise ValueError("anticipation must be a nonnegative integer")
        if base_period not in ("varying", "universal"):
            raise ValueError("base_period must be varying or universal")
        if treatment is None and first_treated is None:
            raise ValueError("Provide treatment or first_treated for StaggeredDiD.")
        if control_group not in ("never_treated", "not_yet_treated"):
            raise ValueError("control_group must be 'never_treated' or 'not_yet_treated'")
        if type(b_reps) is not int or b_reps < 1:
            raise ValueError("b_reps must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        super().__init__(data=data, outcome=outcome, covariates=covariates, cluster=cluster, alpha=alpha)
        self.raw_data = data if isinstance(data, pd.DataFrame) else data.to_pandas()
        self.unit_col, self.time_col = unit, time
        self.treatment_col, self.first_treated_col = treatment, first_treated
        self.control_group, self.b_reps, self.seed = control_group, b_reps, seed
        self.anticipation, self.base_period, self.est_method = anticipation, base_period, est_method

    def _prepare(self) -> tuple[np.ndarray, np.ndarray, SmartPeriodIndex, int, np.ndarray | None]:
        df = self.raw_data.copy()
        roles = [self.unit_col, self.time_col, self.outcome]
        if self.treatment_col:
            roles.append(self.treatment_col)
        if self.first_treated_col:
            roles.append(self.first_treated_col)
        if self.cluster:
            roles.append(self.cluster)
        missing = set(roles) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if df.empty or df[[self.unit_col, self.time_col, self.outcome]].isna().any().any():
            raise ValueError("StaggeredDiD requires nonmissing unit, time and outcome values")
        if df.duplicated([self.unit_col, self.time_col]).any():
            raise ValueError("StaggeredDiD requires unique unit-time observations; duplicates found")
        index, t = SmartPeriodIndex.from_series(df[self.time_col])
        df["_q_period"] = t
        n_units, n_periods = df[self.unit_col].nunique(), len(index.labels)
        if len(df) != n_units * n_periods or n_periods < 3:
            raise ValueError("StaggeredDiD requires a balanced panel with at least 3 periods; missing unit-time periods")
        try:
            outcomes = pd.to_numeric(df[self.outcome], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("StaggeredDiD outcomes must be numeric and finite") from exc
        if not np.isfinite(outcomes).all():
            raise ValueError("StaggeredDiD outcomes must be numeric and finite")

        if self.treatment_col:
            if df[self.treatment_col].isna().any() or not df[self.treatment_col].isin([0, 1, False, True]).all():
                raise ValueError("StaggeredDiD treatment must be a nonmissing binary absorbing indicator")
            path = df.sort_values("_q_period").groupby(self.unit_col)[self.treatment_col]
            if any(np.any(np.diff(s.to_numpy(dtype=int)) < 0) for _, s in path):
                raise ValueError("StaggeredDiD requires absorbing treatment; reversible paths unsupported")
            first_seen = df.loc[df[self.treatment_col].astype(bool)].groupby(self.unit_col)["_q_period"].min()
            cohort = df[self.unit_col].map(first_seen).fillna(-1).astype(int)
        else:
            cohort = pd.Series(-1, index=df.index, dtype=int)

        if self.first_treated_col:
            by_unit = df.groupby(self.unit_col)[self.first_treated_col]
            if any(s.nunique(dropna=False) > 1 for _, s in by_unit):
                raise ValueError("first_treated must be constant for each unit, including missing markers")
            labels = df[self.first_treated_col]
            # R 'did' convention: 0 for never treated when 0 is not a valid panel period.
            sentinels = (-1, 0)
            never_markers = labels.isna()
            for marker in sentinels:
                if marker not in index.time_to_idx:
                    never_markers |= labels.eq(marker)
            unsupported = labels[~never_markers & ~labels.isin(index.time_to_idx)]
            if not unsupported.empty:
                raise ValueError(f"first_treated has adoption times outside observed panel: {unsupported.unique().tolist()[:5]}")
            derived = labels.map(index.time_to_idx).where(~never_markers, -1).astype(int)
            if self.treatment_col and not np.array_equal(cohort.to_numpy(), derived.to_numpy()):
                raise ValueError("Treatment indicator and first_treated cohort are inconsistent")
            cohort = derived

        df["_q_cohort"] = cohort.to_numpy()
        by_cohort = df.groupby(self.unit_col)["_q_cohort"]
        if any(s.nunique() != 1 for _, s in by_cohort):
            raise ValueError("Treatment cohort must be constant per unit")
        # Keep the same sorted unit index for outcome matrix and adoption cohort.
        yframe = df.pivot(index=self.unit_col, columns="_q_period", values=self.outcome)
        gframe = df.drop_duplicates(self.unit_col).set_index(self.unit_col)["_q_cohort"]
        Y = yframe.to_numpy(dtype=float)
        G = gframe.reindex(yframe.index).to_numpy(dtype=int)
        if np.any((G >= 0) & (G <= self.anticipation)):
            raise ValueError("Treated cohort in first observed period or with no pre-treatment baseline given anticipation")
        cluster_labels = None
        if self.cluster is not None:
            if df[self.cluster].isna().any():
                raise ValueError("Cluster identifiers must not be missing")
            by_unit_clusters = df.groupby(self.unit_col)[self.cluster]
            if any(group.nunique(dropna=False) != 1 for _, group in by_unit_clusters):
                raise ValueError("Cluster must be time-invariant within each panel unit")
            ordered_clusters = (pd.Series(yframe.index.to_numpy(), index=yframe.index) if self.cluster == self.unit_col
                                else df.drop_duplicates(self.unit_col).set_index(self.unit_col)[self.cluster].reindex(yframe.index))
            cluster_labels = pd.factorize(ordered_clusters, sort=False)[0]
        if not np.any(G > 0):
            raise ValueError("No treated cohorts have an identifiable pre-treatment period")
        if self.control_group == "never_treated" and not np.any(G == -1):
            raise ValueError("No never-treated units; choose control_group='not_yet_treated'")
        if self.est_method in ("dr", "dr_improved"):
            for col in self.covariates:
                if col not in df.columns:
                    raise ValueError(f"Missing DR covariate: {col}")
                if not pd.api.types.is_numeric_dtype(df[col]) or not np.isfinite(df[col].to_numpy(dtype=float)).all():
                    raise ValueError(f"DR covariate {col} must be finite numeric")
        return Y, G, index, len(df), cluster_labels

    def fit(self) -> StaggeredDiDResult:
        Y, G, period_index, n_obs, cluster_labels = self._prepare()
        N, T = Y.shape
        inf_plan = InferencePlan.build(cluster_labels, N)
        cohorts = sorted(np.unique(G[G > 0]).tolist())
        sizes = {int(g): int(np.sum(G == g)) for g in cohorts}
        effects: dict[tuple[int, int], float] = {}
        influence: dict[tuple[int, int], np.ndarray] = {}
        support: dict[tuple[int, int], dict[str, Any]] = {}
        skipped: list[dict[str, Any]] = []

        X_by_period = None
        if self.est_method in ("dr", "dr_improved"):
            # Covariates are measured at each comparison baseline, never post-treatment.
            # Row order is aligned to the outcome matrix, not input order.
            ordered_units = self.raw_data.pivot(index=self.unit_col, columns=self.time_col, values=self.outcome).index
            raw = self.raw_data.copy()
            raw["_q_period"] = period_index.from_series(raw[self.time_col])[1]
            X_by_period = {t: raw.loc[raw["_q_period"] == t].set_index(self.unit_col).reindex(ordered_units)[self.covariates].to_numpy(dtype=float) for t in range(T)}
        for g in cohorts:
            group = G == g
            n_group = sizes[g]
            for t in range(T):
                clean_base = g - self.anticipation - 1
                if t == clean_base:
                    # The last uncontaminated period is the normalized reference.
                    continue
                # Varying: adjacent clean pretreatment changes; universal:
                # common clean baseline, even for preperiod event comparisons.
                base = (t - 1 if self.base_period == "varying" and t < clean_base
                        else clean_base)
                if base < 0:
                    continue
                if self.control_group == "never_treated":
                    control = G == -1
                else:
                    # A control must be untreated at BOTH comparison dates;
                    # the focal treatment cohort must never be its own control.
                    # Later adopters may already be responding during anticipation.
                    control = ((G == -1) | (G > max(t, base) + self.anticipation)) & ~group
                n_control = int(np.sum(control))
                # At coarser clustering, both comparison arms need replicated
                # independent clusters; two units in one cluster are not two
                # independent treatment or comparison observations.
                n_treated_clusters = int(np.unique(cluster_labels[group]).size) if cluster_labels is not None else n_group
                n_control_clusters = int(np.unique(cluster_labels[control]).size) if cluster_labels is not None else n_control
                if n_group < 2 or n_control < 2 or n_treated_clusters < 2 or n_control_clusters < 2:
                    skipped.append({"cohort": period_index.idx_to_time[g],
                                    "time": period_index.idx_to_time[t],
                                    "is_post": bool(t >= g),
                                    "is_anticipatory": bool(clean_base < t < g),
                                    "reason": "fewer than two treated/control units or independent clusters",
                                    "n_treated": n_group, "n_controls": n_control,
                                    "n_treated_clusters": n_treated_clusters, "n_control_clusters": n_control_clusters})
                    continue
                delta = Y[:, t] - Y[:, base]
                key = (g, t)
                psi = np.zeros(N)
                overlap = {}
                if self.est_method in ("dr", "dr_improved"):
                    eligible = group | control
                    dr_fn = improved_dr_panel if self.est_method == "dr_improved" else traditional_dr_panel
                    cell = dr_fn(delta[eligible], group[eligible], X_by_period[base][eligible])
                    effects[key] = cell.att
                    psi[eligible] = cell.influence * (N / int(eligible.sum()))
                    overlap = {"min_fitted_propensity": cell.min_propensity,
                               "max_fitted_propensity": cell.max_propensity}
                else:
                    treated_mean = np.mean(delta[group])
                    control_mean = np.mean(delta[control])
                    effects[key] = float(treated_mean - control_mean)
                    psi[group] = (delta[group] - treated_mean) * (N / n_group)
                    psi[control] = -(delta[control] - control_mean) * (N / n_control)
                influence[key] = psi
                support[key] = {"n_treated": n_group, "n_controls": n_control,
                                "n_treated_clusters": n_treated_clusters, "n_control_clusters": n_control_clusters,
                                "base_t": base, **overlap}

        post = sorted(pair for pair in effects if pair[1] >= pair[0])
        if not post:
            raise ValueError("No identified post-treatment cohort-time effects with sufficient comparison support")
        weights = np.array([sizes[g] for g, _ in post], dtype=float)
        total_post_weight = float(weights.sum())
        weights /= total_post_weight
        overall = float(np.dot(weights, [effects[k] for k in post]))
        psi_total = np.column_stack([influence[k] for k in post]) @ weights
        # Callaway-Sant'Anna (2021) Theorem 2 unconditional cohort-share IF term:
        # accounts for random cohort shares under heterogeneous treatment effects.
        psi_weight = np.zeros(N, dtype=float)
        for (g, t), w_k in zip(post, weights):
            ind_g = (G == g).astype(float)
            p_g = sizes[g] / N
            psi_weight += ((ind_g - p_g) * (effects[(g, t)] - overall)) * (N / total_post_weight)
        uncond_se = inf_plan.se(psi_total + psi_weight)
        # Preserve sample-conditional overall_se as primary for compatibility while exposing unconditional_se.
        overall_se = inf_plan.se(psi_total)
        if overall_se <= np.finfo(float).eps * max(1.0, abs(overall)):
            raise ValueError("Zero estimated sampling variance; cannot report a meaningful DiD confidence interval")
        p_val = inf_plan.tail_probability(overall / overall_se)
        z = inf_plan.critical(self.alpha)

        event_effect: dict[int, float] = {}
        event_psi: dict[int, np.ndarray] = {}
        event_se: dict[int, float] = {}
        reference_event = -(self.anticipation + 1)
        event_times = sorted({t - g for g, t in effects} | {reference_event})
        for e in event_times:
            if e == reference_event:
                event_effect[e], event_se[e] = 0.0, 0.0
                event_psi[e] = np.zeros(N)
                continue
            keys = sorted(pair for pair in effects if pair[1] - pair[0] == e)
            w = np.array([sizes[g] for g, _ in keys], dtype=float)
            w /= w.sum()
            event_effect[e] = float(np.dot(w, [effects[k] for k in keys]))
            event_psi[e] = np.column_stack([influence[k] for k in keys]) @ w
            event_se[e] = inf_plan.se(event_psi[e])

        # Expose additional transparent views of the identified post cells.
        cohort_effect: dict[int, float] = {}
        cohort_se: dict[int, float] = {}
        calendar_effect: dict[int, float] = {}
        calendar_se: dict[int, float] = {}
        for g in cohorts:
            keys = [k for k in post if k[0] == g]
            if keys:
                w = np.full(len(keys), 1.0 / len(keys))
                cohort_effect[g] = float(np.dot(w, [effects[k] for k in keys]))
                cohort_se[g] = inf_plan.se(np.column_stack([influence[k] for k in keys]) @ w)
        for t in range(T):
            keys = [k for k in post if k[1] == t]
            if keys:
                w = np.array([sizes[g] for g, _ in keys], dtype=float)
                w /= w.sum()
                calendar_effect[t] = float(np.dot(w, [effects[k] for k in keys]))
                calendar_se[t] = inf_plan.se(np.column_stack([influence[k] for k in keys]) @ w)

        # No giant b_reps x N matrix is needed. The max-t statistic is formed
        # in fixed-size chunks; zero-variance entries (including e=-1) have no
        # stochastic t-statistic and are omitted from the uniform-band max.
        stochastic = [e for e in event_times if event_se[e] > 0]
        critical = z
        if stochastic:
            unit_psi = np.column_stack([event_psi[e] for e in stochastic])
            se = np.array([event_se[e] for e in stochastic])
            critical = max(z, inf_plan.multiplier_max_t(unit_psi, se, reps=self.b_reps, seed=self.seed, alpha=self.alpha))
        uniform = {e: (event_effect[e] - critical * event_se[e],
                       event_effect[e] + critical * event_se[e]) for e in event_times}

        diagnostics: dict[str, DiagnosticTest] = {}
        leads = [e for e in event_times if e < reference_event and e in event_psi and event_se[e] > 0]
        if leads:
            v = np.column_stack([event_psi[e] for e in leads])
            covariance = inf_plan.covariance(v)
            rank = int(np.linalg.matrix_rank(covariance))
            if rank == len(leads):
                observed = np.array([event_effect[e] for e in leads])
                wald = float(observed @ np.linalg.solve(covariance, observed))
                if cluster_labels is not None:
                    p = float(stats.f.sf(wald / len(leads), len(leads), inf_plan.n_clusters - 1))
                else:
                    p = float(stats.chi2.sf(wald, df=len(leads)))
                status = DiagnosticStatus.PASS if p >= .10 else (DiagnosticStatus.WARN if p >= .05 else DiagnosticStatus.FAIL)
                diagnostics["Pre-Trends Joint Test"] = DiagnosticTest(
                    name="Unconditional pre-period placebo differences", status=status,
                    statistic_name="Wald F (experimental clustered)" if cluster_labels is not None else "Wald Chi2",
                    statistic_value=wald / len(leads) if cluster_labels is not None else wald, p_value=p,
                    threshold="p >= 0.10", message=(
                        "No detectable difference in observed adjacent-period placebo changes; "
                        "this does not establish parallel trends." if p >= .10 else
                        "Pre-period placebo changes show evidence of divergent trends."),
                    remedy=None if p >= .10 else "Inspect pretreatment periods and design assumptions",
                )
            else:
                diagnostics["Pre-Trends Joint Test"] = DiagnosticTest(
                    name="Unconditional pre-period placebo differences",
                    status=DiagnosticStatus.WARN, statistic_name="not_estimable_rank_deficient",
                    statistic_value=0.0, p_value=None, threshold="full-rank placebo covariance",
                    message="Joint Wald statistic not computed because the placebo covariance is singular",
                )

        event_cov = inf_plan.covariance(np.column_stack([event_psi[e] for e in event_times]))
        return StaggeredDiDResult(
            estimate=overall, se=overall_se, p_value=p_val,
            ci=(overall - z * overall_se, overall + z * overall_se),
            outcome_name=self.outcome, n_obs=n_obs,
            n_treated=int(np.sum(G > 0)),
            event_study_effects=event_effect, event_study_se=event_se,
            simultaneous_ci=uniform,
            group_time_effects=effects,
            group_time_se={k: inf_plan.se(psi) for k, psi in influence.items()},
            cell_support=support, excluded_cells=skipped,
            cohort_sizes=sizes, cohort_effects=cohort_effect, cohort_se=cohort_se,
            calendar_effects=calendar_effect, calendar_se=calendar_se,
            control_group=self.control_group, anticipation=self.anticipation,
            base_period=self.base_period, est_method=self.est_method,
            inference_plan=inf_plan, cluster_col=self.cluster,
            diagnostics=diagnostics, period_index=period_index, underlying_model=self,
            event_study_cov=event_cov, unconditional_se=uncond_se,
        )
