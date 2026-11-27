"""Immutable, explicit statistical designs; parameters never identify assumptions.

Design specs constrain input and routing but do not certify estimator validity.
Only options supported by current experimental backends are exposed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Optional, Tuple, Union

from quasi.errors import InvalidDesignError, UnsupportedInferenceError


def _name(value: str, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidDesignError(f"{key} must be a non-empty column name")


def _unique_columns(*cols: object) -> None:
    names = []
    for value in cols:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, tuple):
            names.extend(value)
    if len(set(names)) != len(names):
        raise InvalidDesignError("The design assigns more than one incompatible role to a column")


@dataclass(frozen=True, kw_only=True)
class DesignSpec:
    outcome: str
    alpha: float = 0.05
    covariates: Tuple[str, ...] = ()
    cluster: Optional[str] = None
    missing_policy: str = "raise"
    family: str = field(default="abstract", init=False)

    def __post_init__(self) -> None:
        _name(self.outcome, "outcome")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (float, int)) or not isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise InvalidDesignError("alpha must be finite and strictly between 0 and 1")
        if not isinstance(self.covariates, (tuple, list)):
            raise InvalidDesignError("covariates must be a tuple or list of column names")
        object.__setattr__(self, "covariates", tuple(self.covariates))
        for col in self.covariates:
            _name(col, "covariate")
        if self.cluster is not None:
            _name(self.cluster, "cluster")
        if self.missing_policy != "raise":
            raise InvalidDesignError("Only missing_policy='raise' is supported; implicit row deletion is forbidden")
        _unique_columns(self.outcome, self.covariates)


@dataclass(frozen=True, kw_only=True)
class PanelDesign(DesignSpec):
    unit: str
    time: str
    treatment: Optional[str] = None
    first_treated: Optional[str] = None
    control_group: str = "never_treated"
    b_reps: int = 9999
    seed: int = 42
    anticipation: int = 0
    base_period: str = "varying"
    est_method: str = "unadjusted"
    family: str = field(default="did", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _name(self.unit, "unit")
        _name(self.time, "time")
        if (self.treatment is None) == (self.first_treated is None):
            raise InvalidDesignError("PanelDesign requires exactly one of treatment or first_treated")
        _name(self.treatment or self.first_treated, "treatment/first_treated")
        if self.control_group not in ("never_treated", "not_yet_treated"):
            raise InvalidDesignError("control_group must be 'never_treated' or 'not_yet_treated'")
        if type(self.b_reps) is not int or self.b_reps < 1:
            raise InvalidDesignError("b_reps must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise InvalidDesignError("seed must be a nonnegative integer")
        _unique_columns(self.outcome, self.covariates, self.unit, self.time, self.treatment, self.first_treated)
        if type(self.anticipation) is not int or self.anticipation < 0:
            raise InvalidDesignError("anticipation must be nonnegative integer")
        if self.base_period not in ("varying", "universal"):
            raise InvalidDesignError("base_period must be varying or universal")
        if self.est_method not in ("unadjusted", "dr", "dr_improved"):
            raise InvalidDesignError("est_method must be unadjusted, dr, or dr_improved")
        if self.covariates and self.est_method not in ("dr", "dr_improved"):
            raise UnsupportedInferenceError("DiD covariates require est_method='dr' or 'dr_improved'")
        if self.est_method in ("dr", "dr_improved") and not self.covariates:
            raise InvalidDesignError("DiD dr/dr_improved requires baseline covariates")


@dataclass(frozen=True, kw_only=True)
class RDDDesign(DesignSpec):
    running_var: str
    cutoff: float
    treatment: Optional[str] = None
    bandwidth: Union[str, float] = "cct_optimal"
    kernel: str = "triangular"
    order: int = 1
    inference: str = "conventional"
    bias_bandwidth: Optional[float] = None
    donut_radius: float = 0.0
    diagnostic_covariates: Tuple[str, ...] = ()
    vce: str = "hc0"
    family: str = field(default="rdd", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _name(self.running_var, "running_var")
        if self.treatment is not None:
            _name(self.treatment, "treatment")
        if not isinstance(self.diagnostic_covariates, (tuple, list)):
            raise InvalidDesignError("diagnostic_covariates must be a tuple/list of column names")
        object.__setattr__(self, "diagnostic_covariates", tuple(self.diagnostic_covariates))
        for name in self.diagnostic_covariates:
            _name(name, "diagnostic_covariate")
        _unique_columns(self.outcome, self.covariates, self.running_var,
                        self.treatment, self.diagnostic_covariates)
        if isinstance(self.donut_radius, bool) or not isinstance(self.donut_radius, (float, int)) or not isfinite(self.donut_radius) or self.donut_radius < 0:
            raise InvalidDesignError("donut_radius must be finite and nonnegative")
        if isinstance(self.cutoff, bool) or not isinstance(self.cutoff, (int, float)) or not isfinite(self.cutoff):
            raise InvalidDesignError("cutoff must be a finite number")
        valid_auto = ("cct_optimal", "heuristic_pilot", "mserd", "cerrd", "rdrobust_mserd", "rdrobust_cerrd")
        if self.bandwidth not in valid_auto and (isinstance(self.bandwidth, bool) or not isinstance(self.bandwidth, (int, float)) or not isfinite(self.bandwidth) or self.bandwidth <= 0):
            raise InvalidDesignError("bandwidth must be a positive number or a supported selector: " + ", ".join(valid_auto))
        if isinstance(self.bandwidth, (float, int)) and not isinstance(self.bandwidth, bool) and self.donut_radius >= self.bandwidth:
            raise InvalidDesignError("donut_radius must be smaller than the bandwidth")
        if self.vce not in ("hc0", "hc1", "hc2", "hc3", "nn"):
            raise InvalidDesignError("RDD vce must be one of hc0, hc1, hc2, hc3, nn")
        if self.kernel not in ("triangular", "uniform", "epanechnikov"):
            raise InvalidDesignError("Unknown RDD kernel")
        if type(self.order) is not int or self.order not in (1, 2):
            raise InvalidDesignError("RDD polynomial order must be 1 or 2")
        if self.inference not in ("conventional", "bias_corrected"):
            raise InvalidDesignError("RDD inference must be conventional or bias_corrected")
        if self.bias_bandwidth is not None:
            if (isinstance(self.bias_bandwidth, bool) or not isinstance(self.bias_bandwidth, (float, int))
                or not isfinite(self.bias_bandwidth) or self.bias_bandwidth <= 0):
                raise InvalidDesignError("bias_bandwidth must be positive and finite")
            if self.inference != "bias_corrected":
                raise InvalidDesignError("bias_bandwidth requires bias_corrected inference")
        if self.inference == "bias_corrected" and self.cluster is not None:
            raise UnsupportedInferenceError("Clustered bias-corrected RD inference is not validated")
        if self.covariates:
            raise UnsupportedInferenceError("RDD covariate adjustment is not implemented")
        if self.cluster is not None and self.treatment is not None:
            raise UnsupportedInferenceError("Clustered fuzzy RDD inference is not implemented")


@dataclass(frozen=True, kw_only=True)
class IVDesign(DesignSpec):
    endogenous: str
    instruments: Tuple[str, ...]
    covariance: Optional[str] = None
    family: str = field(default="iv", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _name(self.endogenous, "endogenous")
        if not isinstance(self.instruments, (tuple, list)) or not self.instruments:
            raise InvalidDesignError("IVDesign requires at least one instrument")
        object.__setattr__(self, "instruments", tuple(self.instruments))
        for col in self.instruments:
            _name(col, "instrument")
        _unique_columns(self.outcome, self.endogenous, self.instruments, self.covariates)
        allowed = ("CR0", "CR1") if self.cluster is not None else ("HC0", "HC1")
        if self.covariance is not None and self.covariance not in allowed:
            raise InvalidDesignError("IV covariance must be HC0/HC1 without cluster or CR0/CR1 with cluster")


@dataclass(frozen=True, kw_only=True)
class ITSDesign(DesignSpec):
    time: str
    intervention_date: Any
    hac_lag: Optional[int] = None
    error_model: str = "hac"
    ar1_max_iter: int = 100
    ar1_tolerance: float = 1e-7
    seasonal_period: Optional[int] = None
    seasonal_harmonics: int = 1
    family: str = field(default="its", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _name(self.time, "time")
        _unique_columns(self.outcome, self.time, self.covariates)
        if self.intervention_date is None:
            raise InvalidDesignError("intervention_date must be specified")
        if self.hac_lag is not None and (type(self.hac_lag) is not int or self.hac_lag < 0):
            raise InvalidDesignError("hac_lag must be a nonnegative integer")
        if self.error_model not in ("hac", "hac_pw", "ewc", "ar1", "ar1_mle", "ar1_reml"):
            raise InvalidDesignError("error_model must be hac, hac_pw, ewc, ar1, ar1_mle, or ar1_reml")
        if self.error_model in ("ar1", "ar1_mle", "ar1_reml", "ewc") and self.hac_lag is not None:
            raise InvalidDesignError("hac_lag is unavailable with AR1 or EWC model-based inference")
        if self.error_model in ("ar1_mle", "ar1_reml") and self.ar1_max_iter != 100:
            raise InvalidDesignError("ar1_max_iter applies only to error_model='ar1'; use ar1_tolerance for ar1_mle/ar1_reml")
        if type(self.ar1_max_iter) is not int or not 2 <= self.ar1_max_iter <= 10000:
            raise InvalidDesignError("ar1_max_iter must be an integer between 2 and 10000")
        if isinstance(self.ar1_tolerance, bool) or not isinstance(self.ar1_tolerance, (float, int)) or not isfinite(self.ar1_tolerance) or not 0 < self.ar1_tolerance < 0.1:
            raise InvalidDesignError("ar1_tolerance must be finite and between 0 and 0.1")
        if self.seasonal_period is None:
            if self.seasonal_harmonics != 1:
                raise InvalidDesignError("seasonal_harmonics requires seasonal_period")
        else:
            if type(self.seasonal_period) is not int or self.seasonal_period < 2:
                raise InvalidDesignError("seasonal_period must be an integer >= 2")
            if type(self.seasonal_harmonics) is not int or not 1 <= self.seasonal_harmonics <= self.seasonal_period//2:
                raise InvalidDesignError("seasonal_harmonics must be between 1 and floor(period/2)")
        if self.cluster is not None:
            raise UnsupportedInferenceError("cluster inference for ITS is not implemented")


@dataclass(frozen=True, kw_only=True)
class SyntheticDesign(DesignSpec):
    unit: str
    time: str
    treated_unit: Any
    intervention_time: Any
    n_placebos: int = 50
    seed: int = 42
    estimator: str = "sdid"
    treatment: str | None = None
    sparsify: str = "none"
    family: str = field(default="sdid", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _name(self.unit, "unit")
        _name(self.time, "time")
        if self.treatment is not None:
            _name(self.treatment, "treatment")
            if self.treatment in {self.outcome, self.unit, self.time, *self.covariates}:
                raise InvalidDesignError("Synthetic treatment indicator cannot duplicate another design role")
        _unique_columns(self.outcome, self.unit, self.time, self.covariates)
        if self.treated_unit is None or self.intervention_time is None:
            raise InvalidDesignError("SyntheticDesign requires treated_unit and intervention_time")
        if type(self.n_placebos) is not int or self.n_placebos < 0:
            raise InvalidDesignError("n_placebos must be a nonnegative integer")
        if not isinstance(self.treated_unit, str) or not self.treated_unit:
            raise InvalidDesignError("treated_unit must be a nonempty string label (current experimental backend converts labels to str)")
        if type(self.seed) is not int or self.seed < 0:
            raise InvalidDesignError("seed must be a nonnegative integer")
        if self.estimator not in ("sdid", "scm"):
            raise InvalidDesignError("SyntheticDesign.estimator must be 'sdid' or 'scm'")
        if self.sparsify not in ("none", "default"):
            raise InvalidDesignError("SyntheticDesign.sparsify must be 'none' or 'default'")
        if self.estimator != "sdid" and self.sparsify != "none":
            raise InvalidDesignError("sparsify applies only to estimator='sdid'")
        object.__setattr__(self, "family", self.estimator)
        if self.cluster is not None or self.covariates:
            raise UnsupportedInferenceError("Synthetic methods clustering and covariate adjustment are not implemented")


@dataclass(frozen=True, kw_only=True)
class ObservationalDesign(DesignSpec):
    treatment: str
    estimator: str = "aipw"
    caliper: float | None = None
    n_neighbors: int = 1
    n_splits: int = 1
    fold_seed: int = 42
    stabilized: bool = False
    family: str = field(default="aipw_experimental", init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        _name(self.treatment, "treatment")
        _unique_columns(self.outcome, self.treatment, self.covariates)
        if not self.covariates:
            raise InvalidDesignError("ObservationalDesign requires pretreatment covariates")
        if self.estimator not in ("aipw", "psm", "ipw"):
            raise InvalidDesignError("ObservationalDesign.estimator must be aipw, ipw, or psm")
        if self.estimator != "aipw" and self.stabilized:
            raise InvalidDesignError("stabilized applies only to estimator='aipw'")
        if self.estimator != "psm" and self.caliper is not None:
            raise InvalidDesignError("caliper only applies to propensity matching")
        if type(self.n_neighbors) is not int or self.n_neighbors < 1:
            raise InvalidDesignError("n_neighbors must be a positive integer")
        if self.estimator != "psm" and self.n_neighbors != 1:
            raise InvalidDesignError("n_neighbors only applies to propensity matching")
        _cross_splits = self.n_splits
        _cross_seed = self.fold_seed
        if type(_cross_splits) is not int or not 1 <= _cross_splits <= 20:
            raise InvalidDesignError("n_splits must be an integer in [1,20]")
        if type(_cross_seed) is not int or not 0 <= _cross_seed <= (2**32 - 1):
            raise InvalidDesignError("fold_seed must be a nonnegative 32-bit integer")
        if self.estimator == "psm" and self.n_splits != 1:
            raise InvalidDesignError("Cross-fitting is not supported for nearest-neighbor matching")
        if self.caliper is not None and (isinstance(self.caliper, bool) or not isinstance(self.caliper, (int, float)) or not isfinite(self.caliper) or not 0 < self.caliper <= 1):
            raise InvalidDesignError("caliper must be in (0,1]")
        object.__setattr__(self, "family", {"aipw": "aipw_experimental",
                                              "psm": "psm_experimental",
                                              "ipw": "ipw_experimental"}[self.estimator])
        if self.cluster is not None:
            raise UnsupportedInferenceError("Clustered observational inference is not implemented")
