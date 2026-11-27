"""Phase 4 design, data, serialization, and migration contracts.

These tests do not establish statistical parity or identification.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import numpy as np
import pandas as pd
import pytest

import quasi
from quasi.results import ResultRecord


def iv_frame(seed=17, n=100):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 2 * z + rng.normal(size=n)
    return pd.DataFrame({"y": 1.2 * x + rng.normal(size=n), "x": x, "z": z, "w": rng.normal(size=n)})


def its_frame(n=40, start=20):
    t = np.arange(n)
    rng = np.random.default_rng(31)
    return pd.DataFrame({"t": t, "y": 2 + .3 * t + 2 * (t >= start) + rng.normal(0, .2, n)})


def did_frame():
    rng = np.random.default_rng(11)
    rows = []
    for i in range(16):
        for t in range(4):
            d = int(i < 8 and t >= 2)
            rows.append((i, t, d, 2 + t * .1 + d * 2 + rng.normal(0, .05)))
    return pd.DataFrame(rows, columns=["id", "t", "d", "y"])


def rd_frame(n=200):
    rng = np.random.default_rng(12)
    x = rng.uniform(-1, 1, n)
    return pd.DataFrame({"x": x, "y": 2 + x + (x >= 0) * 1.2 + rng.normal(0, .1, n)})


def sdid_frame():
    rows = []
    for u in ("a", "b", "c", "d"):
        for t in range(8):
            rows.append((u, t, 5 + 0.5*t + (2 if u == "a" and t >= 4 else 0) + .01*ord(u)))
    return pd.DataFrame(rows, columns=["unit", "period", "y"])


@pytest.mark.parametrize("spec", [
    lambda: quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",)),
    lambda: quasi.ITSDesign(outcome="y", time="t", intervention_date=20),
    lambda: quasi.RDDDesign(outcome="y", running_var="x", cutoff=0),
    lambda: quasi.PanelDesign(outcome="y", unit="id", time="t", treatment="d"),
    lambda: quasi.SyntheticDesign(outcome="y", unit="unit", time="period", treated_unit="a", intervention_time=4),
    lambda: quasi.ObservationalDesign(outcome="y", treatment="d", covariates=("x",)),
])
def test_immutable_named_specifications(spec):
    design = spec()
    assert design.family != "abstract"
    with pytest.raises(FrozenInstanceError):
        design.alpha = 0.99


@pytest.mark.parametrize("bad,fragment", [
    ({"instruments": ("z", "z")}, "incompatible"),
    ({"instruments": ()}, "instrument"),
    ({"instruments": ("x",)}, "incompatible"),
    ({"covariance": "CR1"}, "covariance"),
    ({"alpha": 0}, "alpha"),
    ({"alpha": float("nan")}, "alpha"),
    ({"missing_policy": "drop"}, "missing_policy"),
    ({"outcome": ""}, "outcome"),
])
def test_invalid_iv_design_fails_early(bad, fragment):
    options = dict(outcome="y", endogenous="x", instruments=("z",))
    options.update(bad)
    with pytest.raises((quasi.InvalidDesignError, quasi.UnsupportedInferenceError), match=fragment):
        quasi.IVDesign(**options)


@pytest.mark.parametrize("options", [
    {"running_var": "x", "cutoff": 0, "instruments": ["z"]},
    {"unit": "id", "time": "t", "treatment": "d", "intervention_date": 2},
    {"unit": "id", "time": "t", "treatment": "d", "treated_unit": "id0"},
])
def test_old_keyword_router_rejects_mixed_family(options):
    with pytest.raises(quasi.InvalidDesignError, match="Ambiguous|Unsupported"):
        quasi.estimate(pd.DataFrame({"y": [1]}), "y", **options)


def test_explicit_design_and_legacy_iv_estimates_agree():
    df = iv_frame()
    explicit = quasi.estimate(df, design=quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",)))
    legacy = quasi.estimate(df, "y", endogenous="x", instruments=["z"])
    named = quasi.estimate(df, "y", method="iv", endogenous="x", instruments=["z"])
    assert explicit.estimate == pytest.approx(legacy.estimate)
    assert explicit.estimate == pytest.approx(named.estimate)
    assert explicit._design_audit.n_rows == len(df)
    assert explicit._design_spec.family == "iv"


def test_explicit_outcome_conflict_and_kwargs_refused():
    df = iv_frame()
    spec = quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",))
    with pytest.raises(quasi.InvalidDesignError, match="conflicts"):
        quasi.estimate(df, outcome="wrong", design=spec)
    with pytest.raises(quasi.InvalidDesignError, match="all options"):
        quasi.estimate(df, design=spec, kernel="triangular")
    with pytest.raises(quasi.InvalidDesignError, match="Ambiguous|Unsupported options"):
        quasi.estimate(df, "y", endogenous="x", instruments=["z"], cutoff=0)


def test_iv_design_rejects_missing_infinite_and_no_variation():
    spec = quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",))
    df = iv_frame()
    df.loc[0, "z"] = np.nan
    with pytest.raises(quasi.InvalidDesignError, match="Missing values"):
        quasi.validate_design_data(spec, df)
    df.loc[0, "z"] = np.inf
    with pytest.raises(quasi.InvalidDesignError, match="nonfinite"):
        quasi.validate_design_data(spec, df)
    df["z"] = 0
    with pytest.raises(quasi.InsufficientSupportError, match="variation"):
        quasi.validate_design_data(spec, df)


def test_duplicate_unbalanced_and_reversed_panel_are_blocked():
    spec = quasi.PanelDesign(outcome="y", unit="id", time="t", treatment="d")
    df = did_frame()
    assert quasi.validate_design_data(spec, df).checks[-1] == "absorbing_treatment_support"
    with pytest.raises(quasi.InvalidDesignError, match="Duplicate"):
        quasi.validate_design_data(spec, pd.concat([df, df.iloc[[0]]]))
    with pytest.raises(quasi.InvalidDesignError, match="Unbalanced"):
        quasi.validate_design_data(spec, df.iloc[:-1])
    df.loc[(df.id == 0) & (df.t == 3), "d"] = 0
    with pytest.raises(quasi.InvalidDesignError, match="Reversible"):
        quasi.validate_design_data(spec, df)


def test_early_adoption_rejected_not_silently_excluded():
    spec = quasi.PanelDesign(outcome="y", unit="id", time="t", treatment="d")
    df = did_frame()
    df.loc[df.id == 0, "d"] = 1
    with pytest.raises(quasi.NotIdentifiedError, match="First observed period"):
        quasi.estimate(df, design=spec)


def test_rdd_has_two_sided_support_and_bandwidth_support():
    spec = quasi.RDDDesign(outcome="y", running_var="x", cutoff=0, bandwidth=.001)
    with pytest.raises(quasi.InsufficientSupportError, match="bandwidth"):
        quasi.validate_design_data(spec, rd_frame())
    with pytest.raises(quasi.InsufficientSupportError, match="both sides"):
        quasi.validate_design_data(quasi.RDDDesign(outcome="y", running_var="x", cutoff=2), rd_frame())


def test_its_regular_time_and_samples_enforced():
    spec = quasi.ITSDesign(outcome="y", time="t", intervention_date=20)
    df = its_frame()
    assert "regular_time_cadence" in quasi.validate_design_data(spec, df).checks
    df["t"] = df["t"].astype(float)
    df.loc[df.t == 5, "t"] = 5.25
    with pytest.raises(quasi.InvalidDesignError, match="regular"):
        quasi.validate_design_data(spec, df)
    df = its_frame()
    df.loc[df.t == 5, "t"] = 4
    with pytest.raises(quasi.InvalidDesignError, match="unique"):
        quasi.validate_design_data(spec, df)


def test_its_monthly_calendar_cadence_and_gap():
    dates = pd.date_range("2022-01-01", periods=40, freq="MS")
    df = its_frame().assign(t=dates)
    spec = quasi.ITSDesign(outcome="y", time="t", intervention_date=dates[20])
    assert quasi.validate_design_data(spec, df).n_rows == 40
    df.loc[5, "t"] = dates[5] + pd.Timedelta(days=2)
    with pytest.raises(quasi.InvalidDesignError, match="regular"):
        quasi.validate_design_data(spec, df)


def test_synthetic_support_and_panel_keys():
    df = sdid_frame()
    spec = quasi.SyntheticDesign(outcome="y", unit="unit", time="period", treated_unit="a", intervention_time=4)
    assert quasi.validate_design_data(spec, df).n_units == 4
    with pytest.raises(quasi.InsufficientSupportError, match="treated_unit"):
        quasi.validate_design_data(quasi.SyntheticDesign(outcome="y", unit="unit", time="period", treated_unit="absent", intervention_time=4), df)


def test_canonical_iv_record_roundtrip_and_is_detached():
    df = iv_frame()
    r = quasi.estimate(df, design=quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",)))
    record = r.to_record()
    assert isinstance(record, ResultRecord)
    assert record.schema_version == "1.0-draft"
    assert record.inference.method == "IV_2SLS_HC0"
    assert record.inference.independently_validated is False
    assert record.identification_status == "not_established"
    assert record.provenance.method_spec_json
    obj = json.loads(r.to_json())
    assert obj["sample"]["input_rows"] == 100
    assert obj["sample"]["analyzed_rows"] == 100
    assert obj["provenance"]["method_spec"]["instruments"] == ["z"]
    assert obj["experimental"] is True
    assert ResultRecord.from_json(r.to_json()).to_json() == r.to_json()
    df.loc[0, "y"] = 99999
    assert ResultRecord.from_json(r.to_json()) == record
    assert len(r.to_frame()) == 1
    assert set(["ci_lower", "confidence_level", "identification_status"]) <= set(r.to_frame())


def test_canonical_its_record_confidence_and_assumptions():
    df = its_frame()
    r = quasi.estimate(df, design=quasi.ITSDesign(outcome="y", time="t", intervention_date=20, alpha=.10))
    record = r.to_record()
    assert record.inference.confidence_level == pytest.approx(.90)
    assert len(record.assumptions) >= 2
    assert all(a.evidence_status == "not_established" for a in record.assumptions)
    assert ResultRecord.from_json(r.to_json()).to_dict() == record.to_dict()
    assert len(r.diagnostic_table()) >= 1
    assert r.to_metadata_dict()["inference_validation"] == "pending"


def test_roundtrip_rejects_fake_identification_or_fake_statistical_approval():
    r = quasi.estimate(iv_frame(), design=quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",)))
    payload = json.loads(r.to_json())
    payload["identification_status"] = "established"
    with pytest.raises(ValueError, match="identification"):
        ResultRecord.from_json(json.dumps(payload))
    payload["identification_status"] = "not_established"
    payload["inference"]["independently_validated"] = True
    with pytest.raises(ValueError, match="independent inference validation"):
        ResultRecord.from_json(json.dumps(payload))


def test_legacy_direct_constructor_reports_missing_preflight():
    r = quasi.TwoStageLS(iv_frame(), "y", "x", ["z"]).fit()
    assert r.to_record().sample.missing_policy == "not_recorded"
    assert r.to_record().provenance.method_spec_json == "{}"


def test_existing_unambiguous_its_rdd_did_routing():
    r1 = quasi.estimate(its_frame(), "y", time="t", intervention_date=20)
    r2 = quasi.estimate(rd_frame(), "y", running_var="x", cutoff=0)
    r3 = quasi.estimate(did_frame(), "y", unit="id", time="t", treatment="d", b_reps=20)
    assert all(np.isfinite([r1.estimate, r2.estimate, r3.estimate]))
    assert r3._design_spec.family == "did"
    for result in (r1, r2, r3):
        assert result.to_record().experimental is True


def test_numeric_looking_string_outcome_rejected_before_fit():
    spec = quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",))
    df = iv_frame()
    df["y"] = df["y"].astype(str)
    with pytest.raises(quasi.InvalidDesignError, match="numeric dtype"):
        quasi.estimate(df, design=spec)


def test_mixed_cohort_missingness_rejected():
    df = did_frame()
    df["cohort"] = np.where(df["id"] < 8, 2., np.nan)
    df.loc[(df["id"] == 0) & (df["t"] == 0), "cohort"] = np.nan
    design = quasi.PanelDesign(outcome="y", unit="id", time="t", first_treated="cohort")
    with pytest.raises(quasi.InvalidDesignError, match="constant per unit"):
        quasi.validate_design_data(design, df)


def test_preflight_does_not_disqualify_fuzzy_rdd_using_global_means():
    df = rd_frame(400)
    df["d"] = 0
    # Equal overall means can arise even when local treatment differences matter.
    x = df["x"].to_numpy()
    below = np.where(x < 0)[0]
    above = np.where(x >= 0)[0]
    df.loc[below[:10], "d"] = 1
    df.loc[above[:10], "d"] = 1
    audit = quasi.validate_design_data(quasi.RDDDesign(outcome="y", running_var="x", cutoff=0, treatment="d"), df)
    assert any("not established" in w for w in audit.warnings)


def test_constructors_reject_unsupported_seeds_and_synth_labels():
    with pytest.raises(quasi.InvalidDesignError, match="seed"):
        quasi.PanelDesign(outcome="y", unit="id", time="t", treatment="d", seed=-1)
    with pytest.raises(quasi.InvalidDesignError, match="treated_unit"):
        quasi.SyntheticDesign(outcome="y", unit="unit", time="period", treated_unit=1, intervention_time=4)


def test_rdd_did_sdid_result_serialization():
    rdd = quasi.estimate(rd_frame(), design=quasi.RDDDesign(outcome="y", running_var="x", cutoff=0))
    did = quasi.estimate(did_frame(), design=quasi.PanelDesign(outcome="y", unit="id", time="t", treatment="d", b_reps=20))
    sdid = quasi.estimate(sdid_frame(), design=quasi.SyntheticDesign(outcome="y", unit="unit", time="period", treated_unit="a", intervention_time=4, n_placebos=3))
    for r in (rdd, did, sdid):
        obj = json.loads(r.to_json())
        assert obj["experimental"] is True
        assert obj["identification_status"] == "not_established"
        assert ResultRecord.from_json(r.to_json()).to_json() == r.to_json()
