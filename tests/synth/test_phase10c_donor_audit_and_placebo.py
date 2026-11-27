"""Iteration 21: verifiable donor exposure and explicitly uncalibrated placebo variance."""
import json

import numpy as np
import pandas as pd
import pytest

import quasi
from quasi.errors import InvalidDesignError, ConvergenceError
from quasi.synth import sdid
from quasi.synth.panel import prepare_synthetic_panel


def panel_data(*, donor_count=5, onset=6, n_periods=12):
    rng = np.random.default_rng(740)
    times = np.arange(n_periods)
    shared = .22 * times + np.sin(times / 2)
    donor_outcomes = np.array([shared + j * .9 + rng.normal(0, .03, n_periods)
                               for j in range(donor_count)])
    treated = 4 + .3 * donor_outcomes[0] + .7 * donor_outcomes[1] + 2 * (times >= onset)
    rows = [("target", int(t), float(y), int(t >= onset)) for t, y in enumerate(treated)]
    rows.extend((f"d{j}", int(t), float(y), 0)
                for j, outcomes in enumerate(donor_outcomes)
                for t, y in enumerate(outcomes))
    return pd.DataFrame(rows, columns=["unit", "period", "y", "exposed"])


@pytest.mark.parametrize("constructor", [quasi.SyntheticDiD, quasi.SyntheticControl])
def test_direct_audit_verified_when_full_observed_pattern_supplied(constructor):
    df = panel_data()
    result = constructor(df, "y", "unit", "period", "target", 6,
                         treatment="exposed", n_placebos=0).fit()
    assert result.treatment_audit["status"] == "observed_pattern_checked"
    assert result.model_meta["treatment_audit"]["donor_treatment_exposure_checked"]
    assert result.treatment_audit["spillovers_verified"] is False
    assert result.model_meta["treatment_audit"]["treatment_column"] == "exposed"
    assert json.loads(result.to_json())["additional_metadata"]["model_meta"]["treatment_audit"]["status"] == "observed_pattern_checked"


@pytest.mark.parametrize("name", ["sdid", "scm"])
def test_explicit_design_routes_audit(name):
    df = panel_data()
    spec = quasi.SyntheticDesign(outcome="y", unit="unit", time="period",
        treated_unit="target", intervention_time=6, treatment="exposed",
        estimator=name, n_placebos=0)
    r = quasi.estimate(df, design=spec)
    assert r.treatment_audit["status"] == "observed_pattern_checked"
    assert "observed_synthetic_treatment_history" in r._design_audit.checks


@pytest.mark.parametrize("name", ["sdid", "scm"])
def test_donor_exposure_refused(name):
    df = panel_data()
    df.loc[(df.unit == "d0") & (df.period == 7), "exposed"] = 1
    constructor = quasi.SyntheticControl if name == "scm" else quasi.SyntheticDiD
    with pytest.raises(InvalidDesignError, match="donor.*untreated"):
        constructor(df, "y", "unit", "period", "target", 6,
                    treatment="exposed", n_placebos=0).fit()
    spec = quasi.SyntheticDesign(outcome="y", unit="unit", time="period",
        treated_unit="target", intervention_time=6, treatment="exposed", estimator=name)
    with pytest.raises(InvalidDesignError, match="donor.*untreated"):
        quasi.estimate(df, design=spec)


@pytest.mark.parametrize("unit,time,value", [
    ("target", 3, 1), ("target", 6, 0), ("target", 11, 0), ("target", 0, 1)])
def test_treated_onset_pattern_refused(unit, time, value):
    df = panel_data()
    df.loc[(df.unit == unit) & (df.period == time), "exposed"] = value
    with pytest.raises(InvalidDesignError, match="Treated unit treatment history"):
        quasi.SyntheticDiD(df, "y", "unit", "period", "target", 6,
                           treatment="exposed", n_placebos=0).fit()


@pytest.mark.parametrize("value", [2, -1, .25, "yes", None])
def test_nonbinary_or_missing_treatment_refused(value):
    df = panel_data()
    df["exposed"] = df["exposed"].astype(object)
    df.loc[(df.unit == "d1") & (df.period == 3), "exposed"] = value
    with pytest.raises(InvalidDesignError, match="binary|nonmissing|missing"):
        quasi.SyntheticDiD(df, "y", "unit", "period", "target", 6,
                           treatment="exposed", n_placebos=0).fit()


def test_missing_treatment_column_and_invalid_duplicate_field_refused():
    df = panel_data()
    with pytest.raises(InvalidDesignError, match="missing|Missing"):
        quasi.SyntheticDiD(df, "y", "unit", "period", "target", 6, treatment="ghost").fit()
    with pytest.raises(InvalidDesignError, match="distinct"):
        quasi.SyntheticDiD(df, "y", "unit", "period", "target", 6, treatment="y").fit()
    with pytest.raises(InvalidDesignError, match="duplicate"):
        quasi.SyntheticDesign(outcome="y", unit="unit", time="period", treated_unit="target",
                              intervention_time=6, treatment="y")


def test_unobserved_history_explicitly_unknown():
    df = panel_data().drop(columns="exposed")
    for cls in [quasi.SyntheticControl, quasi.SyntheticDiD]:
        r = cls(df, "y", "unit", "period", "target", 6, n_placebos=0).fit()
        assert r.treatment_audit["status"] == "not_observed"
        assert not r.treatment_audit["donor_treatment_exposure_checked"]
        assert r.inference_details["reference_certified"] is False


def test_experimental_placebo_is_seeded_and_primary_inference_remains_missing():
    r = quasi.SyntheticDiD(panel_data(), "y", "unit", "period", "target", 6,
                           treatment="exposed", n_placebos=0).fit()
    result1 = r.experimental_placebo_variance(replications=25, seed=19)
    result2 = r.experimental_placebo_variance(replications=25, seed=19)
    assert result1 == result2
    assert result1["status"] == "experimental_descriptive_only"
    assert result1["spread_candidate"] > 0
    assert result1["variance_candidate"] == pytest.approx(result1["spread_candidate"]**2)
    assert result1["sampling_inference_certified"] is False
    assert r.inference_details["method"] == "not_available"
    assert np.isnan(r.se) and np.isnan(r.ci_lower) and np.isnan(r.p_value)


def test_reproducible_resampling_matches_independently_recomputed_scale():
    frame = panel_data(donor_count=5)
    model = quasi.SyntheticDiD(frame, "y", "unit", "period", "target", 6,
                                treatment="exposed", n_placebos=0)
    result = model.fit()
    actual = result.experimental_placebo_variance(replications=30, seed=133)
    p = prepare_synthetic_panel(frame, outcome="y", unit="unit", time="period",
        treated_unit="target", intervention_time=6, treatment="exposed")
    rng = np.random.default_rng(133)
    estimates = []
    for _ in range(30):
        sampled = rng.permutation(5)
        target = p.donors[sampled[-1]]
        comparison = p.donors[sampled[:-1]]
        w, lam, _ = sdid._sdid_weights(target, comparison, 6)
        estimates.append(sdid._sdid_effect(target, comparison, 6, w, lam))
    independent = (29 / 30) * np.var(estimates, ddof=1)
    assert actual["variance_candidate"] == pytest.approx(independent, abs=1e-10)


@pytest.mark.parametrize("reps, seed", [(19, 1), (True, 1), (20, -1), (20, True)])
def test_invalid_resampling_configuration_fails_closed(reps, seed):
    result = quasi.SyntheticDiD(panel_data(), "y", "unit", "period", "target", 6,
                                n_placebos=0).fit()
    with pytest.raises(InvalidDesignError):
        result.experimental_placebo_variance(replications=reps, seed=seed)


def test_insufficient_control_units_rejected_for_candidate_variance():
    result = quasi.SyntheticDiD(panel_data(donor_count=2), "y", "unit", "period",
                                "target", 6, n_placebos=0).fit()
    with pytest.raises(InvalidDesignError, match="3 observed untreated"):
        result.experimental_placebo_variance(replications=20)


def test_failure_to_fit_placebo_never_publishes_candidate(monkeypatch):
    result = quasi.SyntheticDiD(panel_data(), "y", "unit", "period", "target", 6,
                                n_placebos=0).fit()
    def fail(*args, **kwargs):
        raise ConvergenceError("test failure")
    monkeypatch.setattr(sdid, "_sdid_weights", fail)
    with pytest.raises(ConvergenceError, match="replicate"):
        result.experimental_placebo_variance(replications=25)
