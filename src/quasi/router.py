"""Explicit design API with a guarded legacy-keyword migration path.

Keyword routing is retained for pre-1.0 source compatibility but only when one
unambiguous family can be inferred. Explicit ``design=`` is recommended.
"""
from __future__ import annotations

from typing import Any

from quasi.designs import (
    DesignSpec, PanelDesign, RDDDesign, IVDesign, ITSDesign,
    SyntheticDesign, ObservationalDesign, validate_design_data,
)
from quasi.errors import InvalidDesignError

_ALLOWED = {
    "did": (PanelDesign, {"unit", "time", "treatment", "first_treated", "control_group", "b_reps", "seed", "anticipation", "base_period", "est_method"}),
    "rdd": (RDDDesign, {"running_var", "cutoff", "treatment", "bandwidth", "kernel", "order", "inference", "bias_bandwidth", "donut_radius", "diagnostic_covariates", "vce"}),
    "iv": (IVDesign, {"endogenous", "instruments", "covariance"}),
    "its": (ITSDesign, {"time", "intervention_date", "hac_lag", "error_model", "ar1_max_iter", "ar1_tolerance", "seasonal_period", "seasonal_harmonics"}),
    "sdid": (SyntheticDesign, {"unit", "time", "treated_unit", "intervention_time", "n_placebos", "seed", "estimator", "treatment", "sparsify"}),
    "scm": (SyntheticDesign, {"unit", "time", "treated_unit", "intervention_time", "n_placebos", "seed", "estimator", "treatment", "sparsify"}),
    "aipw_experimental": (ObservationalDesign, {"treatment", "estimator", "caliper", "n_neighbors", "n_splits", "fold_seed", "stabilized"}),
    "psm": (ObservationalDesign, {"treatment", "estimator", "caliper", "n_neighbors", "n_splits", "fold_seed", "stabilized"}),
    "ipw": (ObservationalDesign, {"treatment", "estimator", "caliper", "n_neighbors", "n_splits", "fold_seed", "stabilized"}),
}
_COMMON = {"covariates", "cluster", "alpha", "missing_policy"}


def _infer_family(options: dict[str, Any]) -> str:
    # Non-None entries signal a family even if a second required field is missing.
    special = {
        "rdd": ("running_var", "cutoff", "bandwidth", "kernel", "order", "donut_radius", "diagnostic_covariates"),
        "iv": ("endogenous", "instruments"),
        "its": ("intervention_date", "hac_lag", "error_model", "ar1_max_iter", "ar1_tolerance", "seasonal_period", "seasonal_harmonics"),
        "sdid": ("treated_unit", "intervention_time", "n_placebos"),
        "did": ("first_treated", "control_group", "b_reps", "anticipation", "base_period", "est_method"),
    }
    families = [fam for fam, keys in special.items() if any(k in options for k in keys)]
    if len(families) > 1:
        raise InvalidDesignError(f"Ambiguous design keywords from {families}; pass one explicit design= spec")
    if families:
        if families[0] == "sdid" and "treatment" in options and "intervention_time" not in options:
            raise InvalidDesignError("Ambiguous treatment history and treated_unit without synthetic intervention_time; supply explicit SyntheticDesign")
        return families[0]
    if "unit" in options and "time" in options and "treatment" in options:
        return "did"
    if "treatment" in options and "covariates" in options and "unit" not in options:
        raise InvalidDesignError("Ambiguous treatment/covariates: pass ObservationalDesign explicitly")
    raise InvalidDesignError("Cannot determine design safely; supply design=PanelDesign/RDDDesign/IVDesign/ITSDesign/SyntheticDesign")


def _dispatch(data: Any, design: DesignSpec):
    audit = validate_design_data(design, data)
    common = dict(data=data, outcome=design.outcome, alpha=design.alpha, covariates=list(design.covariates) or None)
    if isinstance(design, PanelDesign):
        from quasi.did.staggered import StaggeredDiD
        result = StaggeredDiD(**common, unit=design.unit, time=design.time, treatment=design.treatment,
            first_treated=design.first_treated, control_group=design.control_group,
            b_reps=design.b_reps, seed=design.seed, anticipation=design.anticipation,
            base_period=design.base_period, est_method=design.est_method, cluster=design.cluster).fit()
    elif isinstance(design, RDDDesign):
        from quasi.rdd.regression import Discontinuity
        result = Discontinuity(**common, running_var=design.running_var, cutoff=design.cutoff,
            treatment=design.treatment, bandwidth=design.bandwidth, kernel=design.kernel,
            order=design.order, cluster=design.cluster, inference=design.inference,
            bias_bandwidth=design.bias_bandwidth, donut_radius=design.donut_radius,
            diagnostic_covariates=list(design.diagnostic_covariates), vce=design.vce).fit()
    elif isinstance(design, IVDesign):
        from quasi.iv.tsls import TwoStageLS
        result = TwoStageLS(**common, endogenous=design.endogenous, instruments=list(design.instruments),
            cluster=design.cluster, covariance=design.covariance).fit()
    elif isinstance(design, ITSDesign):
        from quasi.its.segmented import InterruptedTimeSeries
        result = InterruptedTimeSeries(**common, time=design.time,
            intervention_date=design.intervention_date, hac_lag=design.hac_lag,
            error_model=design.error_model, ar1_max_iter=design.ar1_max_iter,
            ar1_tolerance=design.ar1_tolerance, seasonal_period=design.seasonal_period,
            seasonal_harmonics=design.seasonal_harmonics).fit()
    elif isinstance(design, SyntheticDesign):
        if design.estimator == "scm":
            from quasi.synth.scm import SyntheticControl as SyntheticEstimator
            result = SyntheticEstimator(**common, unit=design.unit, time=design.time,
                treated_unit=design.treated_unit, intervention_time=design.intervention_time,
                n_placebos=design.n_placebos, seed=design.seed, treatment=design.treatment).fit()
        else:
            from quasi.synth.sdid import SyntheticDiD as SyntheticEstimator
            result = SyntheticEstimator(**common, unit=design.unit, time=design.time,
                treated_unit=design.treated_unit, intervention_time=design.intervention_time,
                n_placebos=design.n_placebos, seed=design.seed, treatment=design.treatment,
                sparsify=design.sparsify).fit()
    elif isinstance(design, ObservationalDesign):
        from quasi.matching.psm import PropensityScoreMatching, AIPWATT, IPWATT
        if design.estimator == "ipw":
            result = IPWATT(**common, treatment=design.treatment, n_splits=design.n_splits, fold_seed=design.fold_seed).fit()
        elif design.estimator == "psm":
            result = PropensityScoreMatching(**common, treatment=design.treatment, caliper=design.caliper, n_neighbors=design.n_neighbors).fit()
        else:
            result = AIPWATT(**common, treatment=design.treatment, n_splits=design.n_splits, fold_seed=design.fold_seed, stabilized=design.stabilized).fit()
    else:
        raise InvalidDesignError("Unsupported design family")
    result._design_audit = audit
    result._design_spec = design
    return result


def estimate(data: Any, outcome: str | None = None, *, design: DesignSpec | None = None,
             method: str | None = None, **kwargs: Any):
    """Estimate a selected experimental method; prefer explicit ``design=``.

    The deprecated keyword-only routing path is allowed for unambiguous calls,
    and refuses mixed-family or unknown arguments rather than choosing precedence.
    """
    if design is not None:
        if not isinstance(design, DesignSpec):
            raise InvalidDesignError("design= must be a DesignSpec instance")
        if method is not None or kwargs:
            raise InvalidDesignError("When design= is supplied, put all options inside the spec (no other keywords)")
        if outcome is not None and outcome != design.outcome:
            raise InvalidDesignError("outcome conflicts with design.outcome")
        return _dispatch(data, design)
    if outcome is None:
        raise InvalidDesignError("outcome must be supplied without design=")
    if method is not None:
        aliases = {"did": "did", "rdd": "rdd", "iv": "iv", "its": "its", "sdid": "sdid", "scm": "scm", "aipw": "aipw_experimental", "psm": "psm", "ipw": "ipw"}
        if method not in aliases:
            raise InvalidDesignError(f"Unknown method={method!r}; accepted: {sorted(aliases)}")
        family = aliases[method]
    else:
        family = _infer_family(kwargs)
    klass, allowed = _ALLOWED[family]
    unexpected = set(kwargs) - allowed - _COMMON
    if unexpected:
        raise InvalidDesignError(f"Unsupported options for {family}: {sorted(unexpected)}; refusing to ignore them")
    try:
        if family == "ipw":
            if kwargs.get("estimator", "ipw") != "ipw":
                raise InvalidDesignError("method='ipw' conflicts with estimator")
            kwargs["estimator"] = "ipw"
        if family == "psm":
            if kwargs.get("estimator", "psm") != "psm":
                raise InvalidDesignError("method='psm' conflicts with estimator")
            kwargs["estimator"] = "psm"
        if family == "aipw_experimental" and kwargs.get("estimator", "aipw") != "aipw":
            raise InvalidDesignError("method='aipw' conflicts with estimator")
        if family == "scm":
            if kwargs.get("estimator", "scm") != "scm":
                raise InvalidDesignError("method='scm' conflicts with estimator setting")
            kwargs["estimator"] = "scm"
        if family == "sdid" and method == "sdid":
            if kwargs.get("estimator", "sdid") != "sdid":
                raise InvalidDesignError("method='sdid' conflicts with estimator setting")
            kwargs["estimator"] = "sdid"
        spec = klass(outcome=outcome, **kwargs)
    except TypeError as err:
        raise InvalidDesignError(f"Incomplete or invalid {family} design: {err}") from err
    return _dispatch(data, spec)
