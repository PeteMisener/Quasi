"""Versioned, serializable statistical result evidence contract (draft v1).

A frozen evidence record is detached from mutable estimator internals. Unknown
inference methods remain 'unverified', and empirical checks never claim causal
identification. These records are structured metadata, not raw private data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = "1.0-draft"
_STATUS = {"not_run", "pass_as_specified", "warn", "fail", "inconclusive", "not_applicable"}


def _json_safe(value: Any) -> Any:
    """Produce strict JSON values, keeping nonfinite diagnostics explicit as null."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Metadata-only: timestamps and Periods may be stored as clear display strings.
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Metadata of type {type(value).__name__} cannot be safely serialized")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(f"Result field {label!r} must be finite; estimator has not produced a valid reportable result")
    return float(value)


@dataclass(frozen=True)
class InferenceRecord:
    method: str
    confidence_level: float
    se: float | None
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    interval_kind: str = "not_verified"
    independently_validated: bool = False

    def __post_init__(self):
        if not self.method or not isinstance(self.method, str):
            raise ValueError("Inference method must be described")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be strictly between zero and one")
        values = (self.se, self.p_value, self.ci_lower, self.ci_upper)
        if self.method == "not_available" and all(value is None for value in values):
            pass  # Explicit point-only result: no invented precision.
        else:
            for field_name in ("se", "p_value", "ci_lower", "ci_upper"):
                _finite_number(getattr(self, field_name), field_name)
            if self.se < 0 or not 0 <= self.p_value <= 1 or self.ci_lower > self.ci_upper:
                raise ValueError("Invalid uncertainty: SE/p-value/interval bounds out of range")
        if self.independently_validated:
            raise ValueError("No Quasi estimator has passed independent inference validation")


@dataclass(frozen=True)
class SampleFlow:
    input_rows: int
    analyzed_rows: int
    excluded_rows: int
    legacy_treated_count: int
    n_units: int | None = None
    n_periods: int | None = None
    missing_policy: str = "not_recorded"

    def __post_init__(self):
        if min(self.input_rows, self.analyzed_rows, self.excluded_rows, self.legacy_treated_count) < 0:
            raise ValueError("Sample counts cannot be negative")
        if self.input_rows != self.analyzed_rows + self.excluded_rows:
            raise ValueError("Sample flow must reconcile analyzed, excluded and input rows")


@dataclass(frozen=True)
class DiagnosticRecord:
    name: str
    status: str
    message: str
    statistic_name: str | None = None
    statistic_value: float | None = None
    p_value: float | None = None
    threshold: str | None = None

    def __post_init__(self):
        if self.status not in _STATUS:
            raise ValueError(f"Unknown diagnostic status: {self.status}")


@dataclass(frozen=True)
class AssumptionRecord:
    statement: str
    evidence_status: str = "not_established"

    def __post_init__(self):
        if self.evidence_status != "not_established":
            raise ValueError("Data diagnostics alone cannot establish causal assumptions")


@dataclass(frozen=True)
class ProvenanceRecord:
    library: str
    distribution_version: str
    backend: str = "native_experimental"
    seed: int | None = None
    method_spec_json: str = "{}"


@dataclass(frozen=True)
class ResultRecord:
    schema_version: str
    model_name: str
    estimand: str
    outcome: str
    estimate: float
    inference: InferenceRecord
    sample: SampleFlow
    diagnostics: tuple[DiagnosticRecord, ...]
    assumptions: tuple[AssumptionRecord, ...]
    provenance: ProvenanceRecord
    identification_status: str = "not_established"
    experimental: bool = True
    additional_metadata_json: str = "{}"

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported result schema {self.schema_version!r}; expected {SCHEMA_VERSION}")
        _finite_number(self.estimate, "estimate")
        if self.identification_status != "not_established" or self.experimental is not True:
            raise ValueError("Experimental results cannot claim proven identification")
        if not self.model_name or not self.estimand or not self.outcome:
            raise ValueError("Result must state model, estimand and outcome")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["provenance"]["method_spec"] = json.loads(value["provenance"].pop("method_spec_json"))
        value["additional_metadata"] = json.loads(value.pop("additional_metadata_json"))
        return value

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, source: str) -> "ResultRecord":
        obj = json.loads(source, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON constant {x}")))
        if not isinstance(obj, dict):
            raise ValueError("Result JSON must be an object")
        if obj.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unknown result schema version")
        try:
            inf = InferenceRecord(**obj["inference"])
            sample = SampleFlow(**obj["sample"])
            diags = tuple(DiagnosticRecord(**item) for item in obj["diagnostics"])
            assumptions = tuple(AssumptionRecord(**item) for item in obj["assumptions"])
            prov = obj["provenance"]
            provenance = ProvenanceRecord(
                library=prov["library"], distribution_version=prov["distribution_version"],
                backend=prov["backend"], seed=prov.get("seed"),
                method_spec_json=json.dumps(prov["method_spec"], sort_keys=True, allow_nan=False),
            )
            return cls(schema_version=obj["schema_version"], model_name=obj["model_name"],
                estimand=obj["estimand"], outcome=obj["outcome"], estimate=obj["estimate"],
                inference=inf, sample=sample, diagnostics=diags, assumptions=assumptions,
                provenance=provenance, identification_status=obj["identification_status"],
                experimental=obj["experimental"],
                additional_metadata_json=json.dumps(obj["additional_metadata"], sort_keys=True, allow_nan=False))
        except (KeyError, TypeError) as err:
            raise ValueError(f"Malformed result record: {err}") from err


_ASSUMPTIONS = {
    "did": ("Parallel trends for eligible comparisons", "No anticipation", "Stable treatment/no interference"),
    "rdd": ("Continuity of potential outcomes at cutoff", "No precise manipulation of running variable", "Local relevance and monotonicity if fuzzy"),
    "iv": ("Instrument relevance", "Exclusion restriction", "Instrument exogeneity", "Monotonicity for a LATE interpretation"),
    "its": ("No simultaneous intervention confounding", "Correct underlying counterfactual trend", "Residual dependence adequately modeled"),
    "sdid": ("Credible donor pool and counterfactual trend", "No spillovers", "Stable preintervention structure"),
    "scm": ("Untreated donor pool throughout the evaluation horizon", "No treatment spillovers", "Stable synthetic counterfactual fit"),
    "aipw_experimental": ("Conditional exchangeability", "Treatment overlap/positivity", "Consistent treatment definition"),
    "ipw_experimental": ("Correct propensity model specification", "Conditional exchangeability", "Positivity", "Consistency"),
    "psm_experimental": ("Conditional exchangeability", "Positivity and comparable matched controls", "Consistency"),
}


def from_estimator_result(result: Any) -> ResultRecord:
    """Convert a legacy QEDResult without claiming more than its metadata proves."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        distribution_version = version("quasi-qed")
    except PackageNotFoundError:
        distribution_version = "0+unknown"
    spec = getattr(result, "_design_spec", None)
    audit = getattr(result, "_design_audit", None)
    if spec is None:
        # Direct constructors bypassed the explicit-design validation preflight.
        family = "unknown_direct_constructor"
        method_spec = {}
    else:
        family = spec.family
        from dataclasses import asdict
        method_spec = _json_safe(asdict(spec))
    sample = SampleFlow(
        input_rows=audit.n_rows if audit is not None else int(result.n_obs),
        analyzed_rows=int(result.n_obs),
        excluded_rows=(audit.n_rows - int(result.n_obs)) if audit is not None else 0,
        legacy_treated_count=int(result.n_treated),
        n_units=audit.n_units if audit is not None else None,
        n_periods=audit.n_periods if audit is not None else None,
        missing_policy="raise" if audit is not None else "not_recorded",
    )
    states = {"PASS": "pass_as_specified", "WARN": "warn", "FAIL": "fail"}
    diags = tuple(DiagnosticRecord(
        name=t.name, status=states.get(t.status.value, "inconclusive"), message=t.message,
        statistic_name=t.statistic_name,
        statistic_value=_json_safe(t.statistic_value),
        p_value=_json_safe(t.p_value), threshold=t.threshold,
    ) for t in result.diagnostics.values())
    return ResultRecord(
        schema_version=SCHEMA_VERSION, model_name=result.model_name,
        estimand=result.estimand, outcome=result.outcome_name,
        estimate=_finite_number(result.estimate, "estimate"),
        inference=InferenceRecord(method=str(getattr(result, "inference_details", {}).get("method", "unverified_estimator_specific")), confidence_level=result.confidence_level,
            se=_json_safe(result.se), p_value=_json_safe(result.p_value),
            ci_lower=_json_safe(result.ci_lower), ci_upper=_json_safe(result.ci_upper)),
        sample=sample, diagnostics=diags,
        assumptions=tuple(AssumptionRecord(v) for v in _ASSUMPTIONS.get(family, ("Identifying assumptions have not been independently evaluated",))),
        provenance=ProvenanceRecord(library="quasi", distribution_version=distribution_version,
            seed=getattr(spec, "seed", None), method_spec_json=json.dumps(method_spec, allow_nan=False, sort_keys=True)),
        additional_metadata_json=json.dumps(_json_safe({"model_meta": result.model_meta,
            "sub_effects": result.sub_effects,
            "design_audit": audit.to_dict() if audit is not None else None,
            "inference_settings": getattr(result, "inference_details", {"reference_certified": False}),
            "warning": "Statistical reference and causal identification validation pending"}), allow_nan=False, sort_keys=True),
    )
