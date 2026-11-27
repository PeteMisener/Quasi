"""
Experimental sharp and fuzzy local-polynomial RD estimators.
Bandwidth and density checks are provisional heuristics, not verified
CCT robust bias correction or canonical McCrary inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.diagnostics.density import cjm_density_test, mccrary_density_test
from quasi.prep.extractor import extract_column_arrays
from quasi.rdd.bandwidth import select_cct_bandwidth, select_mse_bandwidth
from quasi.rdd.reference_bandwidth import REFERENCE_SELECTORS, select_reference_bandwidth
from quasi.rdd.kernels import KERNEL_REGISTRY
from quasi.rdd.local_inference import fuzzy_rd_ar_test, rd_inference


@dataclass
class RDDRobustnessRow:
    specification: str
    bandwidth: float
    kernel: str
    order: int
    estimate: float
    se: float
    p_value: float
    ci: Tuple[float, float]
    n_effective: int
    donut_radius: float = 0.0
    n_excluded: int = 0


class RDDRobustnessTable:
    """Table container for multi-specification sensitivity sweeps."""

    def __init__(self, rows: List[RDDRobustnessRow], confidence_level: float = 0.95) -> None:
        self.rows = rows
        self.confidence_level = confidence_level

    def show(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="quasi.rdd: Robustness & Sensitivity Grid")
        table.add_column("Specification", style="cyan", no_wrap=True)
        table.add_column("Bandwidth", justify="right")
        table.add_column("Kernel", justify="center")
        table.add_column("Order", justify="center")
        table.add_column("Estimate (τ)", justify="right")
        table.add_column("Std. Error", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column(f"{100*self.confidence_level:g}% CI", justify="center")
        table.add_column("N (Effective)", justify="right")

        for r in self.rows:
            p_style = "bold green" if r.p_value < 0.05 else "yellow"
            table.add_row(
                r.specification,
                f"{r.bandwidth:.3f}",
                r.kernel,
                str(r.order),
                f"{r.estimate:+.4f}",
                f"{r.se:.4f}",
                f"[{p_style}]{r.p_value:.4f}[/{p_style}]",
                f"[{r.ci[0]:.3f}, {r.ci[1]:.3f}]",
                f"{r.n_effective:,}",
            )
        console.print(table)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.rows])


class RDDResult(QEDResult):
    """Result container for Sharp and Fuzzy Regression Discontinuity models."""

    def __init__(
        self,
        estimate: float,
        se: float,
        p_value: float,
        ci: Tuple[float, float],
        cutoff: float,
        bandwidth: float,
        kernel: str,
        order: int,
        is_fuzzy: bool,
        outcome_name: str,
        n_obs: int,
        n_treated: int,
        n_effective: int,
        diagnostics: Dict[str, DiagnosticTest],
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        underlying_model: Optional[BaseQEDModel] = None,
        inference_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            estimate=estimate,
            se=se,
            p_value=p_value,
            ci=ci,
            estimand="Fuzzy RD local IV effect (conditional on assumptions)" if is_fuzzy else "Sharp RD local outcome discontinuity",
            outcome_name=outcome_name,
            model_name="FuzzyRDD" if is_fuzzy else "SharpRDD",
            n_obs=n_obs,
            n_treated=n_treated,
            diagnostics=diagnostics,
            sub_effects={
                "cutoff": cutoff,
                "bandwidth": bandwidth,
                "kernel": kernel,
                "order": order,
                "n_effective": n_effective,
            },
            model_meta={"cutoff": cutoff, "bandwidth": bandwidth},
            underlying_model=underlying_model,
        )
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.order = order
        self.is_fuzzy = is_fuzzy
        self.n_effective = n_effective
        self.inference_details = inference_metadata or {"method": "conventional_hc0", "reference_certified": False}
        self._X = X_raw
        self._Y = Y_raw

    def summary(self) -> str:
        design_str = "Fuzzy RDD (Local Wald ratio)" if self.is_fuzzy else "Sharp RDD (Local polynomial)"
        lines = [
            "[EXPERIMENTAL] Research prototype; reference validation pending.",
            "=" * 70,
            f"{design_str} | Outcome: {self.outcome_name} (Cutoff: c = {self.cutoff:g})",
            "=" * 70,
            f"Local RD Estimate:  {self.estimate:+.4f} (SE: {self.se:.4f}, {self.inference_details['method']})",
            f"p-value:            {self.p_value:.4f}",
            f"{100*self.confidence_level:.0f}% Conf. Interval: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Bandwidth (h):      {self.bandwidth:.4f} (Kernel: {self.kernel}, Order: {self.order})",
            f"Effective Sample:   {self.n_effective:,} / {self.n_obs:,} within bandwidth",
            "-" * 70,
            "Identification Checks & Diagnostics:",
        ]
        for t in self.diagnostics.values():
            p_str = f"p = {t.p_value:.4f}" if t.p_value is not None else "-"
            lines.append(f"  • {t.name:<30} [{t.status.value}] {t.statistic_name} = {t.statistic_value:.4f} ({p_str})")
        lines.append("=" * 70)
        return "\n".join(lines)

    def plot(self, n_bins: int = 30, **kwargs: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        from quasi.visualization.discontinuity import render_discontinuity_figure

        return render_discontinuity_figure(
            X=self._X,
            Y=self._Y,
            cutoff=self.cutoff,
            bandwidth=self.bandwidth,
            estimate=self.estimate,
            se=self.se,
            p_value=self.p_value,
            kernel=self.kernel,
            order=self.order,
            n_bins=n_bins,
            is_fuzzy=self.is_fuzzy,
            alpha=1.0 - self.confidence_level,
            **kwargs,
        )

    def robustness(
        self,
        bandwidth_multipliers: Optional[List[float]] = None,
        kernels: Optional[List[str]] = None,
        orders: Optional[List[int]] = None,
    ) -> RDDRobustnessTable:
        """One-call sensitivity grid across bandwidth multipliers, kernels, and orders."""
        if self._model is None:
            raise RuntimeError("Underlying model reference unavailable for robustness sweep.")

        b_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0] if bandwidth_multipliers is None else bandwidth_multipliers
        k_list = ["triangular", "epanechnikov", "uniform"] if kernels is None else kernels
        o_list = [1, 2] if orders is None else orders
        if not b_mults or not k_list or not o_list:
            raise ValueError("RDD robustness grid must have at least one bandwidth, kernel and order")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0 for v in b_mults):
            raise ValueError("RDD robustness bandwidth multipliers must be finite and positive")
        if any(k not in KERNEL_REGISTRY for k in k_list):
            raise ValueError("Unsupported RDD robustness kernel")
        if any(type(o) is not int or o not in (1, 2) for o in o_list):
            raise ValueError("RDD robustness orders must be 1 or 2")

        rows: List[RDDRobustnessRow] = []

        for mult in b_mults:
            for kern in k_list:
                for ord_val in o_list:
                    bw = self.bandwidth * mult
                    label = f"{mult:.2f}x h ({kern[:3].capitalize()}, p={ord_val})"
                    if mult == 1.0 and kern == self.kernel and ord_val == self.order:
                        label += " [BASELINE]"

                    try:
                        res = self._model._fit_internal(bandwidth=bw, kernel=kern, order=ord_val, run_diagnostics=False)
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"Unsupported RDD sensitivity cell h={bw:g}, kernel={kern}, order={ord_val}: {exc}") from exc
                    rows.append(
                        RDDRobustnessRow(
                            specification=label,
                            bandwidth=bw,
                            kernel=kern,
                            order=ord_val,
                            estimate=res.estimate,
                            se=res.se,
                            p_value=res.p_value,
                            ci=(res.ci_lower, res.ci_upper),
                            n_effective=res.n_effective, donut_radius=res.inference_details["donut_radius"],
                            n_excluded=res.inference_details["n_donut_excluded"],
                        )
                    )

        return RDDRobustnessTable(rows, confidence_level=self.confidence_level)

    def donut_sensitivity(self, radii: Optional[List[float]] = None) -> RDDRobustnessTable:
        """Re-estimate at *fixed* h and kernel after excluding |running-c| <= radius.

        Each row is an exploratory extrapolation sensitivity estimate, not an
        identification test or a multiple-testing-adjusted confidence interval.
        Model data and the baseline result remain unchanged.
        """
        if self._model is None:
            raise RuntimeError("Underlying model unavailable")
        radii = [0.0, 0.05 * self.bandwidth, 0.1 * self.bandwidth] if radii is None else radii
        if not radii:
            raise ValueError("Supply at least one donut radius")
        rows = []
        for radius in radii:
            if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius < 0 or radius >= self.bandwidth:
                raise ValueError("Donut radius must be finite, nonnegative and smaller than bandwidth")
            try:
                result = self._model._fit_internal(bandwidth=self.bandwidth, kernel=self.kernel,
                    order=self.order, run_diagnostics=False, donut_radius=float(radius))
            except ValueError as exc:
                raise ValueError(f"Unsupported donut radius {radius:g}: {exc}") from exc
            rows.append(RDDRobustnessRow(
                specification=f"Donut |x-c| <= {radius:.6g} (exploratory)",
                bandwidth=self.bandwidth, kernel=self.kernel, order=self.order,
                estimate=result.estimate, se=result.se, p_value=result.p_value,
                ci=(result.ci_lower, result.ci_upper), n_effective=result.n_effective,
                donut_radius=float(radius), n_excluded=result.inference_details["n_donut_excluded"],
            ))
        return RDDRobustnessTable(rows, confidence_level=self.confidence_level)

    def cjm_density(self, *, p: int = 2, bandwidth: Optional[float] = None) -> DiagnosticTest:
        """Cattaneo, Jansson & Ma (2020) local-polynomial CDF density continuity test."""
        return cjm_density_test(self._X, self.cutoff, p=p, bandwidth=bandwidth)

    def ar_test(self, null_effect: float = 0.0) -> dict:
        """Feir, Lemieux & Marmer (2016) weak-IV-robust local AR test for Fuzzy RD."""
        if not self.is_fuzzy or self._model is None:
            raise ValueError("Local AR test applies only to fitted FuzzyRDD models")
        x_c = self._X - self.cutoff
        y_arr = self._Y
        t_arr = self._model.arrays["treatment"].astype(float)
        donut_r = float(self.inference_details.get("donut_radius", 0.0) or 0.0)
        if donut_r > 0.0:
            keep = np.abs(x_c) > donut_r
            x_c, y_arr, t_arr = x_c[keep], y_arr[keep], t_arr[keep]
        b = self.inference_details.get("pilot_bandwidth")
        return fuzzy_rd_ar_test(
            x_c, y_arr, t_arr,
            self.bandwidth, self.kernel, self.order, null_effect=null_effect,
            bias_bandwidth=b, vce=self._model.vce,
        )

    def bibtex(self) -> str:
        from quasi.reporting.provenance import generate_bibtex
        return generate_bibtex("SharpRDD")

    def methods_text(self) -> str:
        design_type = "Fuzzy (instrumental variables)" if self.is_fuzzy else "Sharp"
        return (
            f"Causal effects at the cutoff (c = {self.cutoff}) were estimated using a {design_type} "
            f"local polynomial regression of order {self.order} with a {self.kernel} kernel. "
            f"Donut exclusion radius = {self.inference_details.get('donut_radius', 0.0):.4g} (extrapolating over excluded scores when nonzero). "
            f"Bandwidth h = {self.bandwidth:.3f} was used "
            f"(selection: {self.inference_details.get('bandwidth_selection')}). "
            f"Inference: {self.inference_details['method']}. "
            f"The experimental bias-correction option is not numerically certified against rdrobust; "
            f"an exploratory binned density screen is not a validated McCrary test."
        )


class Discontinuity(BaseQEDModel):
    """
    Sharp and Fuzzy Regression Discontinuity Estimator.

    Parameters:
        data: DataFrame containing observations.
        outcome: Dependent variable column name.
        running_var: Forcing/score variable determining cutoff assignment.
        cutoff: Threshold value c where treatment turns on.
        treatment: Actual treatment take-up column name. If None -> Sharp RDD. If provided -> Fuzzy RDD.
        bandwidth: 'cct_optimal' (default) or explicit float.
        kernel: 'triangular' (MSE-optimal), 'epanechnikov', or 'uniform'.
        order: Local polynomial degree (1 = local linear, 2 = local quadratic).
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, Any],
        outcome: str,
        running_var: str,
        cutoff: float,
        treatment: Optional[str] = None,
        bandwidth: Union[str, float] = "cct_optimal",
        kernel: Literal["triangular", "epanechnikov", "uniform"] = "triangular",
        order: int = 1,
        covariates: Optional[List[str]] = None,
        cluster: Optional[str] = None,
        alpha: float = 0.05,
        inference: Literal["conventional", "bias_corrected"] = "conventional",
        bias_bandwidth: Optional[float] = None,
        donut_radius: float = 0.0,
        diagnostic_covariates: Optional[List[str]] = None,
        vce: Literal["hc0", "hc1", "hc2", "hc3", "nn"] = "hc0",
    ) -> None:
        if vce.lower() not in ("hc0", "hc1", "hc2", "hc3", "nn"):
            raise ValueError("vce must be 'hc0', 'hc1', 'hc2', 'hc3', or 'nn'")
        if isinstance(donut_radius, bool) or not isinstance(donut_radius, (float, int)) or not np.isfinite(donut_radius) or donut_radius < 0:
            raise ValueError("donut_radius must be finite and nonnegative")
        if isinstance(bandwidth, (int, float)) and donut_radius >= bandwidth:
            raise ValueError("donut_radius must be smaller than bandwidth")
        if diagnostic_covariates is not None and not isinstance(diagnostic_covariates, (list, tuple)):
            raise ValueError("diagnostic_covariates must be a list/tuple of column names")
        diagnostic_covariates = list(diagnostic_covariates or [])
        if len(diagnostic_covariates) != len(set(diagnostic_covariates)) or any(not isinstance(v, str) or not v for v in diagnostic_covariates):
            raise ValueError("diagnostic_covariates must contain unique nonempty column names")
        if any(x in {outcome, running_var, treatment, cluster} for x in diagnostic_covariates):
            raise ValueError("Diagnostic covariates must be distinct from outcome/running/treatment/cluster roles")
        if covariates:
            raise NotImplementedError(
                "RDD covariates currently enter diagnostics only and do not adjust the effect; "
                "omit covariates until covariate-adjusted RD is implemented."
            )
        if inference not in ("conventional", "bias_corrected"):
            raise ValueError("RDD inference must be 'conventional' or 'bias_corrected'")
        if bias_bandwidth is not None and (not np.isfinite(bias_bandwidth) or bias_bandwidth <= 0):
            raise ValueError("bias_bandwidth must be finite and positive")
        if bias_bandwidth is not None and inference != "bias_corrected":
            raise ValueError("bias_bandwidth is only valid with inference='bias_corrected'")
        if inference == "bias_corrected" and cluster is not None:
            raise NotImplementedError("Clustered bias-corrected RDD is not validated")
        if isinstance(bandwidth, (int, float)):
            if isinstance(bandwidth, bool) or not np.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError("bandwidth must be finite and positive")
        elif bandwidth not in ("cct_optimal", "heuristic_pilot", "mserd", "cerrd", *REFERENCE_SELECTORS):
            raise ValueError("Unknown bandwidth selector; use explicit h, mserd, cerrd, heuristic_pilot/cct_optimal, rdrobust_mserd or rdrobust_cerrd")
        if kernel.lower() not in KERNEL_REGISTRY:
            raise ValueError("Unsupported RDD kernel")
        if type(order) is not int or order not in (1, 2):
            raise ValueError("Local polynomial degree must be 1 or 2")
        if not np.isfinite(cutoff):
            raise ValueError("cutoff must be finite")
        if cluster is not None and treatment is not None:
            raise NotImplementedError("Cluster-robust fuzzy RDD inference is not implemented or validated.")
        super().__init__(data=data, outcome=outcome, covariates=covariates, cluster=cluster, alpha=alpha)
        self.running_var = running_var
        self.cutoff = float(cutoff)
        self.treatment_col = treatment
        self.bandwidth_spec = bandwidth
        self._reference_selection = None
        self._native_selection = None
        self.kernel = kernel.lower()
        self.order = order
        self.inference = inference
        self.bias_bandwidth = bias_bandwidth
        self.donut_radius = float(donut_radius)
        self.diagnostic_covariates = diagnostic_covariates
        self.vce = vce.lower()

        # Extract running variable, outcome, and treatment
        cols: Dict[str, Union[str, List[str], None]] = {
            "outcome": self.outcome,
            "running_var": self.running_var,
            "treatment": self.treatment_col,
            "covariates": self.covariates if self.covariates else None,
            "cluster": self.cluster,
        }
        self.arrays = extract_column_arrays(data, cols)
        self.diagnostic_arrays = extract_column_arrays(data, {
            name: name for name in diagnostic_covariates
        })
        for name, values in self.diagnostic_arrays.items():
            if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                raise ValueError(f"Diagnostic covariate {name!r} must be finite numeric data")

    def _fit_internal(
        self,
        bandwidth: float,
        kernel: str,
        order: int,
        run_diagnostics: bool = True,
        donut_radius: Optional[float] = None,
    ) -> RDDResult:
        X_all = self.arrays["running_var"]
        Y_all = self.arrays["outcome"]
        T_all = self.arrays.get("treatment", None)
        is_fuzzy = T_all is not None
        radius = self.donut_radius if donut_radius is None else float(donut_radius)
        if not np.isfinite(radius) or radius < 0 or radius >= bandwidth:
            raise ValueError("donut_radius must be finite, nonnegative and smaller than bandwidth")
        keep = (np.abs(X_all - self.cutoff) > radius) if radius > 0 else np.ones(len(X_all), dtype=bool)
        X = X_all[keep]
        Y = Y_all[keep]
        T = T_all[keep] if T_all is not None else None
        x_c = X - self.cutoff
        in_bw = np.abs(x_c) <= bandwidth

        if np.sum(in_bw & (x_c >= 0)) < (order + 2) or np.sum(in_bw & (x_c < 0)) < (order + 2):
            raise ValueError(
                f"Too few observations inside bandwidth window [c - {bandwidth:.3f}, c + {bandwidth:.3f}]. "
                "Widen the bandwidth or verify data density around cutoff."
            )

        cl = self.arrays["cluster"][keep] if self.cluster else None
        # A pilot bandwidth is used only for an explicitly requested experimental
        # higher-order bias-corrected estimator, never silently substituted for h.
        b = (float(self.bias_bandwidth) if self.bias_bandwidth is not None
             else (self._reference_selection.b if self._reference_selection is not None
                   else (self._native_selection.b if self._native_selection is not None
                         else 1.5 * float(bandwidth)))) if self.inference == "bias_corrected" else None
        fit = rd_inference(x_c.astype(float), Y.astype(float), float(bandwidth), kernel,
                           order, treatment=None if T is None else T.astype(float),
                           bias_bandwidth=b, cluster=cl, vce=self.vce)
        tau_estimate = fit["estimate"]
        se_estimate = fit["se"]
        z_score = tau_estimate / se_estimate
        p_val = float(2 * stats.norm.sf(abs(z_score)))
        crit = float(stats.norm.ppf(1 - self.alpha / 2))
        ci = (tau_estimate - crit * se_estimate, tau_estimate + crit * se_estimate)

        diagnostics: Dict[str, DiagnosticTest] = {}
        if run_diagnostics:
            if is_fuzzy:
                jump_se = fit["first_stage_se"]
                first_stage_z = abs(fit["first_stage_jump"]) / jump_se if jump_se and jump_se > 0 else float("nan")
                first_stage_F = first_stage_z**2
                diagnostics["Fuzzy First Stage"] = DiagnosticTest(
                    name="Fuzzy RD First Stage (exploratory)",
                    status=DiagnosticStatus.WARN if not np.isfinite(first_stage_F) or first_stage_F < 10 else DiagnosticStatus.PASS,
                    statistic_name="Local first-stage Wald F (HC0)",
                    statistic_value=float(first_stage_F),
                    p_value=float(2 * stats.norm.sf(first_stage_z)) if np.isfinite(first_stage_z) else None,
                    threshold="F >= 10 (heuristic, not weak-IV robust)",
                    message="A first-stage diagnostic alone does not establish fuzzy RD validity or guarantee valid Wald inference.",
                )
            if np.sum(x_c == 0):
                diagnostics["Cutoff Mass Points"] = DiagnosticTest(
                    name="Cutoff Mass Points", status=DiagnosticStatus.WARN,
                    statistic_name="Observations exactly at cutoff",
                    statistic_value=float(np.sum(x_c == 0)), p_value=None,
                    threshold="review heaping and assignment rule",
                    message="Cutoff ties can affect continuity-based inference; review the running variable.",
                )
            # The running-variable sorting screen must use the *original* data;
            # removing the donut would mechanically create a false density hole.
            diagnostics["Exploratory Density Screen"] = mccrary_density_test(X_all, self.cutoff)
            if radius > 0:
                diagnostics["Donut Extrapolation"] = DiagnosticTest(
                    name="Donut extrapolation (exploratory)", status=DiagnosticStatus.WARN,
                    statistic_name="Excluded near-cutoff observations",
                    statistic_value=float(np.count_nonzero(~keep)), p_value=None,
                    threshold="sensitivity only",
                    message="Excluded scores near cutoff; estimates extrapolate across the removed region, not a causal validation check.",
                )
            from quasi.rdd.diagnostics import support_screen, covariate_continuity
            diagnostics["Running Variable Support"] = support_screen(x_c, bandwidth, kernel, order)
            if self.diagnostic_arrays:
                continuity = covariate_continuity(
                    x_c, {name: values[keep] for name, values in self.diagnostic_arrays.items()},
                    bandwidth, kernel, order, alpha=self.alpha,
                )
                diagnostics.update(continuity)


        return RDDResult(
            estimate=tau_estimate,
            se=se_estimate,
            p_value=p_val,
            ci=ci,
            cutoff=self.cutoff,
            bandwidth=bandwidth,
            kernel=kernel,
            order=order,
            is_fuzzy=is_fuzzy,
            outcome_name=self.outcome,
            n_obs=len(X_all),
            n_treated=int(np.sum(X_all >= self.cutoff)),
            n_effective=fit["n_left_h"] + fit["n_right_h"],
            diagnostics=diagnostics,
            X_raw=X_all,
            Y_raw=Y_all,
            underlying_model=self,
            inference_metadata={
                "method": fit["inference_method"],
                "bandwidth_selection": (
                    "provisional_heuristic_not_CCT" if self.bandwidth_spec in ("cct_optimal", "heuristic_pilot")
                    else (
                        f"native_{self._native_selection.selector}" if self._native_selection is not None
                        else (
                            self._reference_selection.selector + "_delegated_to_rdrobust"
                            if self._reference_selection is not None else "user_specified"
                        )
                    )
                ),
                "bandwidth_reference_version": (self._reference_selection.reference_version
                                                if self._reference_selection is not None else None),
                "bandwidth_reference_vce": (self._reference_selection.variance
                                            if self._reference_selection is not None else None),
                "bandwidth_reference_masspoints": (self._reference_selection.masspoints
                                                   if self._reference_selection is not None else None),
                "reference_bias_bandwidth": (self._reference_selection.b
                                             if self._reference_selection is not None else None),
                "pilot_bandwidth": b, "pilot_order": order + 1 if b is not None else None,
                "conventional_estimate": fit["conventional_estimate"],
                "bias_corrected_estimate": fit["bias_corrected_estimate"],
                "first_stage_jump": fit["first_stage_jump"],
                "first_stage_se": fit["first_stage_se"],
                "bias_corrected_first_stage_jump": fit["bias_corrected_first_stage_jump"],
                "fuzzy_bias_correction": fit["fuzzy_bias_correction"],
                "n_left_h": fit["n_left_h"], "n_right_h": fit["n_right_h"],
                "n_left_b": fit["n_left_b"], "n_right_b": fit["n_right_b"],
                "unique_left": fit["n_unique_left"], "unique_right": fit["n_unique_right"],
                "donut_radius": radius, "n_donut_excluded": int(np.count_nonzero(~keep)),
                "diagnostic_covariates": list(self.diagnostic_covariates),
                "reference_certified": False,
            },
        )

    def fit(self) -> RDDResult:
        """Fit the RDD model using optimal or explicit bandwidth."""
        X = self.arrays["running_var"]
        Y = self.arrays["outcome"]

        # Select optimal bandwidth if requested
        if self.bandwidth_spec in ("cct_optimal", "heuristic_pilot"):
            self._reference_selection = None
            self._native_selection = None
            h_opt, _, _ = select_cct_bandwidth(
                X=X,
                Y=Y,
                cutoff=self.cutoff,
                kernel=self.kernel,
                order=self.order,
            )
        elif self.bandwidth_spec in ("mserd", "cerrd"):
            self._reference_selection = None
            self._native_selection = select_mse_bandwidth(
                X=X,
                Y=Y,
                cutoff=self.cutoff,
                kernel=self.kernel,
                order=self.order,
                selector=self.bandwidth_spec,
            )
            h_opt = self._native_selection.h
        elif self.bandwidth_spec in REFERENCE_SELECTORS:
            self._native_selection = None
            selected = select_reference_bandwidth(
                Y, X, self.cutoff, self.kernel, self.order, self.bandwidth_spec,
                treatment=self.arrays.get("treatment"),
            )
            self._reference_selection = selected
            h_opt = selected.h
        else:
            self._reference_selection = None
            self._native_selection = None
            h_opt = float(self.bandwidth_spec)
        if self.donut_radius >= h_opt:
            raise ValueError("donut_radius must be smaller than the selected bandwidth")

        return self._fit_internal(
            bandwidth=h_opt,
            kernel=self.kernel,
            order=self.order,
            run_diagnostics=True,
        )


class SharpRDD(Discontinuity):
    """Sharp RD; a supplied treatment indicator would imply a different design."""

    def __init__(self, *args: Any, treatment: Optional[str] = None, **kwargs: Any) -> None:
        if treatment is not None:
            raise ValueError("SharpRDD does not accept treatment; use FuzzyRDD for a fuzzy design.")
        super().__init__(*args, treatment=None, **kwargs)


class FuzzyRDD(Discontinuity):
    """Fuzzy RD; requires an actual treatment indicator distinct from assignment."""

    def __init__(self, *args: Any, treatment: Optional[str] = None, **kwargs: Any) -> None:
        if treatment is None:
            raise ValueError("FuzzyRDD requires a treatment column name.")
        super().__init__(*args, treatment=treatment, **kwargs)
