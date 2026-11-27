"""Independent hand-contrast and adverse panel cases for experimental unadjusted DiD.

These are mathematical/behavioral checks, NOT external R did parity.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from quasi import PanelDesign, StaggeredDiD, estimate
from validation.monte_carlo import run_simulation


def panel(*, never=True, seed=313, noise=.17, times=(2001, 2002, 2003, 2004, 2005, 2006)):
    rng = np.random.default_rng(seed)
    units = list(range(24))
    cohorts = {u: (times[2] if u < 8 else (times[4] if u < 16 else None)) for u in units}
    if not never:
        cohorts.update({u: times[4] for u in range(16, 24)})
    rows = []
    for u in units:
        for t_idx, t in enumerate(times):
            g = cohorts[u]
            onset = times.index(g) if g is not None else 10_000
            d = int(t_idx >= onset)
            effect = (2 if u < 8 else 4) * d + 0.4 * max(0, t_idx - onset) * d
            y = 10 + .2 * t_idx + .6 * u + effect + noise * rng.normal()
            rows.append({"unit": u, "time": t, "outcome": y, "treatment": d,
                         "cohort": g if g is not None else 0})
    return pd.DataFrame(rows)


def fit(df, **kwargs):
    return StaggeredDiD(df, "outcome", "unit", "time", treatment="treatment", **({"b_reps": 101} | kwargs)).fit()


def manual_att(df, g, t, *, comparator="never_treated", baseline=None):
    wide = df.pivot(index="unit", columns="time", values="outcome")
    adoption = df.groupby("unit")["cohort"].first()
    dates = list(wide.columns)
    pos = dates.index(g)
    base = dates[pos - 1] if baseline is None and t >= g else baseline
    if base is None:
        base = dates[dates.index(t) - 1]
    target = adoption == g
    controls = adoption == 0 if comparator == "never_treated" else (
        (adoption == 0) | ((adoption > max(t, base)) & (adoption != g))
    )
    delta = wide[t] - wide[base]
    return float(delta[target].mean() - delta[controls].mean())


def test_every_post_cell_matches_explicit_independent_two_by_two_contrast():
    df = panel()
    result = fit(df)
    assert result.n_treated == 16
    assert len(result.group_time_effects) == 8
    for (g_index, t_index), est in result.group_time_effects.items():
        if t_index < g_index:
            continue
        g = result.period_index.idx_to_time[g_index]
        t = result.period_index.idx_to_time[t_index]
        assert est == pytest.approx(manual_att(df, g, t), abs=1e-12)
    weights = np.asarray([result.cohort_sizes[g] for g, t in result.group_time_effects if t >= g])
    post = np.asarray([v for (g, t), v in result.group_time_effects.items() if t >= g])
    assert result.estimate == pytest.approx(np.dot(weights, post) / weights.sum(), abs=1e-12)
    assert result.model_meta["method_validation"] == "experimental_unverified"
    cohort_summary = result.aggregation_frame("cohort")
    cal_summary = result.aggregation_frame("calendar")
    assert cohort_summary["cohort"].tolist() == [2003, 2005]
    assert cal_summary["calendar"].tolist() == [2003, 2004, 2005, 2006]
    for _, row in cohort_summary.iterrows():
        cohort = result.period_index.time_to_idx[row["cohort"]]
        vals = [v for (g, t), v in result.group_time_effects.items() if g == cohort and t >= g]
        assert row["estimate"] == pytest.approx(np.mean(vals))
    with pytest.raises(ValueError, match="dimension"):
        result.aggregation_frame("none")


def test_varying_prebaseline_is_adjacent_period_not_future_preperiod():
    df = panel(noise=0.15)
    # Before any treatment, impose an abrupt change shared by each treated cohort
    # in 2002 that reverses at 2003. A fixed 2004 baseline would blend those jumps.
    df.loc[(df.cohort == 2005) & (df.time == 2002), "outcome"] += 1.5
    res = fit(df)
    g, t = 4, 1  # Cohort 2005, placebo date 2002, base 2001
    assert res.group_time_effects[(g, t)] == pytest.approx(manual_att(df, 2005, 2002, baseline=2001))
    assert res.cell_support[(g, t)]["base_t"] == 0
    assert -1 in res.event_study_effects
    assert res.event_study_effects[-1] == 0.0
    assert res.event_study_se[-1] == 0.0
    assert res.simultaneous_ci[-1] == (0.0, 0.0)


def test_not_yet_treated_sets_include_later_adopters_and_never_treated():
    df = panel()
    r = fit(df, control_group="not_yet_treated")
    # At time 2003 the later cohort (2005) and never-treated controls are usable.
    assert r.cell_support[(2, 2)]["n_controls"] == 16
    assert r.group_time_effects[(2, 2)] == pytest.approx(
        manual_att(df, 2003, 2003, comparator="not_yet_treated"))
    # In 2005 the 2005 adopter cohort is no longer a control for 2003 cohort.
    assert r.cell_support[(2, 4)]["n_controls"] == 8
    assert r.group_time_effects[(2, 4)] == pytest.approx(
        manual_att(df, 2003, 2005, comparator="not_yet_treated"))
    # In 2002, 2003 and 2005 cohorts both untreated; 2003 focal cohort cannot
    # appear in the 2005 cohort's placebo comparison at 2002 once after its adoption.
    assert r.cell_support[(4, 1)]["n_controls"] == 16


def test_not_yet_without_never_reports_missing_last_post_cells():
    df = panel(never=False)
    r = fit(df, control_group="not_yet_treated")
    assert r.control_group == "not_yet_treated"
    assert r.excluded_cells
    assert any(item["is_post"] and item["time"] == 2006 for item in r.excluded_cells)
    assert len(r.group_time_frame().query("is_post")) < 6
    assert r.model_meta["skipped_cell_count"] == len(r.excluded_cells)
    with pytest.raises(ValueError, match="never-treated"):
        fit(df, control_group="never_treated")


def test_first_treated_zero_marker_matches_treatment_derived_cohorts():
    df = panel()
    direct = StaggeredDiD(df, "outcome", "unit", "time", first_treated="cohort", b_reps=101).fit()
    derived = fit(df)
    assert direct.group_time_effects == derived.group_time_effects
    assert direct.estimate == pytest.approx(derived.estimate)
    spec = PanelDesign(outcome="outcome", unit="unit", time="time", first_treated="cohort", b_reps=101)
    explicit = estimate(df, design=spec)
    assert explicit.estimate == pytest.approx(derived.estimate)
    assert explicit.to_record().experimental


def test_missing_marker_and_negative_marker_supported_but_bad_cohorts_rejected():
    df = panel()
    df.loc[df.cohort == 0, "cohort"] = None
    res = StaggeredDiD(df, "outcome", "unit", "time", first_treated="cohort", b_reps=31).fit()
    assert res.n_treated == 16
    df.loc[df.cohort.isna(), "cohort"] = -1
    res2 = StaggeredDiD(df, "outcome", "unit", "time", first_treated="cohort", b_reps=31).fit()
    assert res2.n_treated == 16
    df.loc[df.unit == 0, "cohort"] = 2099
    with pytest.raises(ValueError, match="outside observed panel"):
        StaggeredDiD(df, "outcome", "unit", "time", first_treated="cohort").fit()


def test_cohort_must_be_constant_within_unit():
    df = panel()
    df.loc[(df.unit == 0) & (df.time == 2004), "cohort"] = 2005
    with pytest.raises(ValueError, match="constant"):
        StaggeredDiD(df, "outcome", "unit", "time", first_treated="cohort").fit()
    with pytest.raises(ValueError, match="constant"):
        estimate(df, design=PanelDesign(outcome="outcome", unit="unit", time="time", first_treated="cohort"))


def test_both_treatment_and_first_treated_must_agree():
    df = panel()
    r = StaggeredDiD(df, "outcome", "unit", "time", treatment="treatment", first_treated="cohort", b_reps=31).fit()
    assert r.n_treated == 16
    df.loc[(df.unit == 1), "cohort"] = 2005
    with pytest.raises(ValueError, match="inconsistent"):
        StaggeredDiD(df, "outcome", "unit", "time", treatment="treatment", first_treated="cohort").fit()


def test_unit_order_and_row_shuffle_leave_effects_invariant():
    df = panel()
    orig = fit(df, seed=51)
    shuffled = fit(df.sample(frac=1.0, random_state=23).reset_index(drop=True), seed=51)
    assert orig.group_time_effects == pytest.approx(shuffled.group_time_effects)
    assert orig.event_study_effects == pytest.approx(shuffled.event_study_effects)
    assert orig.simultaneous_ci == shuffled.simultaneous_ci
    assert orig.se == pytest.approx(shuffled.se)


def test_analytical_two_group_did_unit_influence_se():
    df = panel()
    result = fit(df)
    key = (4, 4)
    wide = df.pivot(index="unit", columns="time", values="outcome")
    cohort = df.groupby("unit")["cohort"].first()
    delta = wide[2005] - wide[2004]
    g = delta[cohort == 2005].to_numpy()
    c = delta[cohort == 0].to_numpy()
    expected_se = np.sqrt(np.sum((g - g.mean()) ** 2) / len(g)**2 +
                          np.sum((c - c.mean()) ** 2) / len(c)**2)
    assert result.group_time_se[key] == pytest.approx(expected_se, rel=1e-12)


def test_simultaneous_bands_respond_to_alpha_and_seed_are_reproducible():
    df = panel()
    a = fit(df, alpha=.05, seed=88, b_reps=399)
    b = fit(df, alpha=.1, seed=88, b_reps=399)
    c = fit(df, alpha=.05, seed=88, b_reps=399)
    assert a.simultaneous_ci == c.simultaneous_ci
    for e in a.event_study_effects:
        lo_a, hi_a = a.simultaneous_ci[e]
        lo_b, hi_b = b.simultaneous_ci[e]
        assert lo_a <= lo_b + 1e-12
        assert hi_a >= hi_b - 1e-12
    assert a.confidence_level == pytest.approx(.95)
    assert b.confidence_level == pytest.approx(.90)
    assert a.to_record().inference.method == "unit_level_conditional_influence_function_analytical_normal"
    assert not a.to_record().inference.independently_validated


def test_constant_effect_without_sampling_variance_is_rejected_instead_of_certified():
    with pytest.raises(ValueError, match="Zero estimated sampling variance"):
        fit(panel(noise=0))


def test_unsupported_options_explicitly_fail():
    df = panel()
    with pytest.raises(NotImplementedError, match="covariates"):
        fit(df, covariates=["cohort"])
    # Phase 6C: unit-ID clustering is now supported (but still experimental).
    result = fit(df, cluster="unit")
    assert result.cluster_col == "unit"
    with pytest.raises(ValueError, match="b_reps"):
        StaggeredDiD(df, "outcome", "unit", "time", treatment="treatment", b_reps=0)


def test_result_table_and_json_export_preserve_experimental_status():
    result = fit(panel())
    cells = result.group_time_frame()
    assert set(("cohort", "time", "event_time", "estimate", "n_controls", "base_time")) <= set(cells.columns)
    assert cells.query("is_post").shape[0] == 6
    export = json.loads(result.to_json())
    assert export["experimental"] is True
    assert export["identification_status"] == "not_established"
    assert export["additional_metadata"]["model_meta"]["control_group"] == "never_treated"


def test_did_monte_carlo_runner_is_reproducible_and_tracks_reps():
    a = run_simulation("did_parallel", reps=5, seed=616)
    b = run_simulation("did_parallel", reps=5, seed=616)
    assert a == b
    assert a.completed_reps + a.failed_reps == 5
    assert a.bias is not None
    assert a.status == "experimental_unreviewed"
