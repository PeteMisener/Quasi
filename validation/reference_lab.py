"""Independent real-data numerical reference laboratory.

Reference generation imports statsmodels but NEVER quasi. Runtime comparison
belongs in tests/parity. No generated/synthetic data is a parity source.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
BACKEND_VERSION = "0.14.6"
SCHEMA_VERSION = 1

# Explicit allowlist: no manifest-supplied arbitrary imports or file paths.
DATASETS = {
    "grunfeld_2sls": ("grunfeld", "iv_2sls", 0.05),
    "nile_its_95": ("nile", "its_hac", 0.05),
    "nile_its_90": ("nile", "its_hac", 0.10),
}


def canonical_data_sha256(df: Any) -> str:
    """Checksum complete input data (in declared original row/column order)."""
    payload = df.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_version_tuple(ver: str) -> tuple[int, ...]:
    parts = []
    for piece in ver.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def source_data(dataset: str, *, enforce_version: bool = False):
    import statsmodels
    import statsmodels.api as sm

    if enforce_version and statsmodels.__version__ != BACKEND_VERSION:
        raise RuntimeError(
            f"Reference backend must be statsmodels=={BACKEND_VERSION}; "
            f"found {statsmodels.__version__}. Never silently regenerate goldens."
        )
    if _parse_version_tuple(statsmodels.__version__) < _parse_version_tuple(BACKEND_VERSION):
        raise RuntimeError(
            f"Reference backend requires statsmodels>={BACKEND_VERSION}; "
            f"found {statsmodels.__version__}."
        )
    if dataset not in {"nile", "grunfeld"}:
        raise ValueError(f"Unrecognized real dataset: {dataset}")
    data = getattr(sm.datasets, dataset).load_pandas().data.copy().reset_index(drop=True)
    if data.empty or data.isna().any().any():
        raise ValueError(f"Dataset {dataset} is empty or missing values")
    return data


def _field(value: float, atol: float = 1e-7, rtol: float = 1e-8) -> dict:
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite numerical benchmark")
    return {"reference_value": float(value), "atol": atol, "rtol": rtol, "status": "active"}


def generate_reference(fixture_id: str, *, enforce_version: bool = True) -> dict:
    """Compute goldens exclusively with Statsmodels independent estimators."""
    import statsmodels
    import statsmodels.api as sm

    ds, method, alpha = DATASETS[fixture_id]
    df = source_data(ds, enforce_version=enforce_version)
    dataset_module = getattr(sm.datasets, ds)
    provenance = {
        "original_source_attribution": str(dataset_module.SOURCE).strip(),
        "upstream_copyright_statement": str(dataset_module.COPYRIGHT).strip(),
        "upstream_reference_module": f"statsmodels.datasets.{ds}",
        "loader": f"statsmodels.datasets.{ds}.load_pandas",
        "source": "Statsmodels packaged public research dataset",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sha256_canonical_csv": canonical_data_sha256(df),
        "redistributed": False,
        "license_review": "pending: refer to Statsmodels dataset attribution before redistribution",
        "substantive_identification_claim": False,
    }
    if method == "iv_2sls":
        from statsmodels.sandbox.regression.gmm import IV2SLS

        X = sm.add_constant(df[["value"]], has_constant="add")
        Z = sm.add_constant(df[["capital"]], has_constant="add")
        ref = IV2SLS(df["invest"], X, Z).fit()
        f = sm.OLS(df["value"], Z).fit().fvalue
        fields = {
            "estimate": _field(ref.params["value"], atol=1e-9, rtol=1e-8),
            "n_obs": _field(len(df), atol=0, rtol=0),
            "first_stage_f": _field(f, atol=1e-7, rtol=1e-8),
            "se": {
                "reference_value": float(ref.bse["value"]),
                "status": "blocked",
                "reason": "Statsmodels default IV2SLS uses homoskedastic covariance; Quasi currently reports HC0-like covariance. These are NOT comparable SEs.",
            },
        }
        design = {
            "outcome": "invest", "endogenous": "value", "instruments": ["capital"],
            "covariates": [], "alpha": alpha,
            "identification_note": "Numerical regression check only; instrument exclusion/exogeneity NOT justified.",
        }
        estimator = "statsmodels.sandbox.regression.gmm.IV2SLS"
    else:
        import numpy as np

        T = len(df)
        time = df["year"].to_numpy()
        intervention_date = 1921.0
        onset = int(np.flatnonzero(time == intervention_date)[0])
        t = np.arange(T, dtype=float)
        X = np.column_stack((np.ones(T), t, (t >= onset).astype(float), np.maximum(t - onset, 0)))
        lag = 3
        ref = sm.OLS(df["volume"].to_numpy(), X).fit(
            cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}, use_t=False
        )
        ci = ref.conf_int(alpha=alpha)
        fields = {
            "estimate": _field(ref.params[2]),
            "se": _field(ref.bse[2]),
            "ci_lower": _field(ci[2][0]),
            "ci_upper": _field(ci[2][1]),
            "p_value": _field(ref.pvalues[2]),
            "slope_change": _field(ref.params[3]),
            "slope_se": _field(ref.bse[3]),
            "n_obs": _field(T, atol=0, rtol=0),
        }
        design = {
            "outcome": "volume", "time": "year", "intervention_date": intervention_date,
            "hac_lag": lag, "alpha": alpha, "use_correction": False, "use_t": False,
            "identification_note": "1921 is a numerical pseudointervention only, NOT an asserted causal event.",
        }
        estimator = "statsmodels.regression.linear_model.OLS.fit(cov_type='HAC')"
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "reference_kind": "independent_real_data_numerical",
        "reference": {"package": "statsmodels", "version": statsmodels.__version__, "estimator": estimator},
        "dataset": provenance,
        "method": method,
        "design": design,
        "fields": fields,
        "statistical_review": {"status": "pending", "reviewer": None},
        "limitations": "Numerical parity is NOT inference validation or credible causal identification.",
    }


def read_fixture(fixture_id: str) -> dict:
    if fixture_id not in DATASETS:
        raise ValueError("Fixture ID not allowlisted")
    file = FIXTURE_DIR / f"{fixture_id}.json"
    if not file.is_file():
        raise FileNotFoundError(f"Missing real-data golden: {file}; never substitute synthetic fallback")
    obj = json.loads(file.read_text(encoding="utf-8"))
    if obj.get("schema_version") != SCHEMA_VERSION or obj.get("fixture_id") != fixture_id:
        raise ValueError("Reference fixture schema/ID mismatch")
    if obj.get("reference_kind") != "independent_real_data_numerical":
        raise ValueError("Synthetic/non-independent inputs cannot be parity fixtures")
    if obj.get("reference", {}).get("version") != BACKEND_VERSION:
        raise ValueError("Reference backend mismatch")
    if obj.get("method") != DATASETS[fixture_id][1]:
        raise ValueError("Method mismatch")
    if obj.get("dataset", {}).get("redistributed") is not False:
        raise ValueError("Real research dataset should be loaded, not silently redistributed")
    dataset = DATASETS[fixture_id][0]
    current_data = source_data(dataset)
    if (obj["dataset"]["sha256_canonical_csv"] != canonical_data_sha256(current_data)
            or obj["dataset"]["rows"] != len(current_data)
            or obj["dataset"]["columns"] != list(current_data.columns)):
        raise ValueError("Dataset has changed since reference generation; stop and review")
    if not obj.get("fields") or not any(f.get("status") == "active" for f in obj["fields"].values()):
        raise ValueError("Fixture contains no active independent numerical comparisons")
    for field, spec in obj["fields"].items():
        if spec.get("status") not in {"active", "blocked"}:
            raise ValueError(f"Unrecognized comparison status for {field}")
        if not math.isfinite(float(spec["reference_value"])):
            raise ValueError(f"Nonfinite reference for {field}")
        if spec["status"] == "active" and (spec.get("atol", -1) < 0 or spec.get("rtol", -1) < 0):
            raise ValueError(f"Invalid tolerance for {field}")
        if spec["status"] == "blocked" and not spec.get("reason"):
            raise ValueError(f"Blocked field {field} requires a reason")
    return obj


def check_reference(fixture_id: str) -> dict:
    """Check committed golden against independently rerun reference, no quasi import."""
    import numpy as np

    frozen = read_fixture(fixture_id)
    fresh = generate_reference(fixture_id, enforce_version=False)
    if (
        frozen["design"] != fresh["design"]
        or frozen["reference"]["package"] != fresh["reference"]["package"]
        or frozen["reference"]["estimator"] != fresh["reference"]["estimator"]
    ):
        raise AssertionError(f"Frozen design or independent reference changed: {fixture_id}")
    for field, original in frozen["fields"].items():
        if field not in fresh["fields"] or fresh["fields"][field]["status"] != original["status"]:
            raise AssertionError(f"Reference field contract changed: {fixture_id}.{field}")
        if not np.isclose(original["reference_value"], fresh["fields"][field]["reference_value"],
                          atol=original.get("atol", 1e-9), rtol=original.get("rtol", 1e-8)):
            raise AssertionError(f"Reference golden drifted: {fixture_id}.{field}")
    return frozen


def compare_quasi_result(frozen: dict, result: Any) -> list[dict]:
    import numpy as np

    comparisons = []
    for field, spec in frozen["fields"].items():
        actual = float(getattr(result, field))
        status = "blocked" if spec["status"] == "blocked" else (
            "pass" if np.isclose(actual, spec["reference_value"], atol=spec["atol"], rtol=spec["rtol"])
            else "fail"
        )
        comparisons.append({
            "field": field, "actual": actual, "reference": spec["reference_value"],
            "status": status, "reason": spec.get("reason", ""),
            "atol": spec.get("atol"), "rtol": spec.get("rtol"),
        })
    return comparisons
