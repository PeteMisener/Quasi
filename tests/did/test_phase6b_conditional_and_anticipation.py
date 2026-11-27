"""Hand-defined and adversarial experimental staggered DiD tests.

The oracle for conditional DR's *point* is a separately computed OLS/logit
formula. These are not independently generated R `did` benchmark outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from quasi import PanelDesign, StaggeredDiD, estimate
from quasi.did.conditional import traditional_dr_panel
from tests.did.test_phase6_group_time import panel


def _fit(df, **kwargs):
    return StaggeredDiD(df, outcome="outcome", unit="unit", time="time",
                        first_treated="cohort", b_reps=39, seed=123, **kwargs).fit()


def test_anticipation_changes_baseline_and_preserves_anticipatory_cells():
    df = panel(noise=0.12)
    # Announcement response in year before treatment for cohort 2005.
    df.loc[(df.cohort == 2005) & (df.time == 2004), "outcome"] += 5
    naive = _fit(df)
    protected = _fit(df, anticipation=1)
    g, t = 4, 4
    assert naive.cell_support[(g, t)]["base_t"] == 3
    assert protected.cell_support[(g, t)]["base_t"] == 2
    assert protected.group_time_effects[(g, t)] - naive.group_time_effects[(g, t)] == pytest.approx(5.0, abs=.3)
    assert protected.cell_support[(g, 3)]["base_t"] == 2
    assert protected.group_time_frame().query('is_anticipatory').shape[0] > 0
    assert protected.event_study_effects[-2] == 0
    assert -1 in protected.event_study_effects
    assert protected.anticipation == 1
    assert protected.to_record().experimental


def test_universal_preperiod_is_cumulative_not_adjacent():
    df = panel(noise=.1)
    df.loc[(df.cohort == 2005) & (df.time == 2002), "outcome"] += 3
    varying = _fit(df, base_period="varying")
    universal = _fit(df, base_period="universal")
    assert varying.group_time_effects[(4, 1)] != pytest.approx(universal.group_time_effects[(4, 1)])
    assert {(g, t): v for (g, t), v in varying.group_time_effects.items() if t >= g} == pytest.approx(
           {(g, t): v for (g, t), v in universal.group_time_effects.items() if t >= g})
    assert (4, 0) not in varying.group_time_effects
    assert (4, 0) in universal.group_time_effects
    assert universal.base_period == "universal"


def test_not_yet_controls_exclude_anticipating_cohort():
    df = panel()
    r0 = _fit(df, control_group="not_yet_treated")
    r1 = _fit(df, control_group="not_yet_treated", anticipation=1)
    # 2005 cohort may already anticipate in 2004, so it is not eligible in 2004.
    assert r0.cell_support[(2, 3)]["n_controls"] == 16
    assert r1.cell_support[(2, 3)]["n_controls"] == 8
    # Anticipation window for 2003 group moves its base to 2001.
    assert r1.cell_support[(2, 3)]["base_t"] == 0


def test_cohort_too_early_for_anticipation_rejected_direct_and_design():
    df = panel()
    # First adoption is 2003 (index 2): two anticipation periods leaves no clean base.
    with pytest.raises(ValueError, match="baseline"):
        _fit(df, anticipation=2)
    spec = PanelDesign(outcome="outcome", unit="unit", time="time", first_treated="cohort", anticipation=2, b_reps=39)
    with pytest.raises(ValueError, match="pretreatment baseline"):
        estimate(df, design=spec)


def _conditional_panel(n=600, seed=2468, *, constant_trend=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = expit(-.2 + 1.1 * x)
    treated = rng.binomial(1, p, n).astype(bool)
    rows = []
    shock = rng.normal(0, .4, size=n)
    for u in range(n):
        for t in range(4):
            # Baseline factor correlates treatment membership and outcome trend.
            trend = 0.4 + 1.8 * x[u]
            if not constant_trend:
                trend += .4 * x[u] ** 2
            y = 2 + x[u] + t * trend + shock[u] * t + (2.0 * (t >= 2) if treated[u] else 0)
            rows.append({"unit": u, "time": t, "outcome": y, "cohort": 2 if treated[u] else -1,
                         "x": x[u], "x_post": x[u] + (1000 if t >= 2 else 0)})
    return pd.DataFrame(rows)


def test_dr_adjusts_group_trend_confounding_and_baseline_design_api():
    df = _conditional_panel()
    crude = _fit(df)
    adjusted = _fit(df, covariates=["x"], est_method="dr")
    assert abs(adjusted.estimate - 2) < .15
    assert abs(crude.estimate - 2) > .3
    assert adjusted.est_method == "dr"
    assert adjusted.model_meta["est_method"] == "dr"
    assert np.isfinite(adjusted.se) and adjusted.se > 0
    spec = PanelDesign(outcome="outcome", unit="unit", time="time", first_treated="cohort",
                       covariates=("x",), est_method="dr", b_reps=39, seed=123)
    via = estimate(df, design=spec)
    assert via.estimate == pytest.approx(adjusted.estimate)
    assert not via.to_record().inference.independently_validated


def test_baseline_covariate_snapshot_not_future_covariate():
    df = _conditional_panel(n=350)
    a = _fit(df, covariates=["x"], est_method="dr")
    # Changing a covariate after the comparison baseline must not influence fit.
    df.loc[df.time >= 2, "x"] += 50000
    b = _fit(df, covariates=["x"], est_method="dr")
    assert a.estimate == pytest.approx(b.estimate)
    assert a.se == pytest.approx(b.se)


def test_traditional_dr_matches_independently_constructed_point_formula():
    rng = np.random.default_rng(45)
    n = 500
    x = rng.normal(size=n)
    d = rng.binomial(1, expit(.25 + x)).astype(bool)
    delta = .8 + 1.4*x + 1.6*d + rng.normal(size=n)
    from scipy.optimize import minimize
    X = np.column_stack([np.ones(n), x])
    result = minimize(lambda b: np.sum(np.logaddexp(0, X@b) - d*(X@b)), np.zeros(2),
                      method='BFGS', jac=lambda b: X.T@(expit(X@b)-d),
                      options={'gtol': 1e-6})
    prob = expit(X @ result.x)
    beta = np.linalg.lstsq(X[~d], delta[~d], rcond=None)[0]
    residual = delta - X@beta
    control_odds = (1-d)*prob/(1-prob)
    expected = residual[d].mean() - (control_odds*residual).sum()/control_odds.sum()
    cell = traditional_dr_panel(delta, d, x[:, None])
    assert cell.att == pytest.approx(expected, abs=1e-7)
    assert abs(cell.influence.sum()) < 1e-5


def test_dr_overlap_and_separation_fail_closed():
    x = np.linspace(-10, 10, 100)
    with pytest.raises(ValueError, match="overlap|converge|singular|rank-deficient"):
        traditional_dr_panel(np.linspace(1,2,100), x>0, x[:,None])


def test_unsupported_conditional_settings_fail_explicitly():
    df = _conditional_panel(n=120)
    with pytest.raises(NotImplementedError, match="est_method"):
        _fit(df, covariates=["x"])
    with pytest.raises(ValueError, match="requires nonempty"):
        _fit(df, est_method="dr")
    with pytest.raises(ValueError, match="anticipation"):
        _fit(df, anticipation=-1)
    with pytest.raises(ValueError, match="base_period"):
        _fit(df, base_period="arbitrary")
    with pytest.raises(ValueError, match="est_method"):
        _fit(df, est_method="ipw")
    with pytest.raises(ValueError, match="overlap|converge|singular|rank-deficient"):
        df["x"] = (df.cohort > 0).astype(int) * 10
        _fit(df, covariates=["x"], est_method="dr")


def test_dr_point_survives_nonlinear_outcome_when_propensity_correct():
    rng = np.random.default_rng(182)
    n = 3500
    x = rng.normal(size=n)
    d = rng.binomial(1, expit(-.2 + .5*x), n).astype(bool)
    # Linear working outcome regression cannot represent the quadratic trend.
    delta = 1 + 1.5*x + .8*x*x + 1.8*d + rng.normal(size=n)
    cell = traditional_dr_panel(delta, d, x[:, None])
    assert abs(cell.att-1.8) < .15


def test_dr_point_survives_misspecified_propensity_when_outcome_correct():
    rng = np.random.default_rng(172)
    n = 3500
    x = rng.normal(size=n)
    d = rng.binomial(1, expit(-.4 + .3*x*x), n).astype(bool)
    # Logistic in x is misspecified; untreated changes are exactly linear in x.
    delta = 1 + .8*x + 1.8*d + rng.normal(size=n)
    cell = traditional_dr_panel(delta, d, x[:, None])
    assert abs(cell.att-1.8) < .15


def test_dr_influence_se_tracks_refitted_pair_bootstrap_on_seeded_design():
    rng = np.random.default_rng(112)
    n = 420
    x = rng.normal(size=n)
    d = rng.binomial(1, expit(.1 + .3*x), n).astype(bool)
    delta = 1 + .6*x + 1.4*d + rng.normal(size=n)
    cell = traditional_dr_panel(delta, d, x[:, None])
    analytic = np.linalg.norm(cell.influence) / n
    bootstrap = []
    # Fresh re-estimation of both nuisance models on every pairs-bootstrap draw.
    for _ in range(120):
        draw = rng.integers(0, n, size=n)
        bootstrap.append(traditional_dr_panel(delta[draw], d[draw], x[draw,None]).att)
    empirical = np.std(bootstrap, ddof=1)
    assert abs(analytic - empirical) / empirical < .22


def test_dr_deterministic_seed_and_row_order_with_experimental_inference():
    data = _conditional_panel(n=250)
    a = _fit(data, covariates=['x'], est_method='dr')
    b = _fit(data.sample(frac=1, random_state=12).reset_index(drop=True),
             covariates=['x'], est_method='dr')
    assert a.estimate == pytest.approx(b.estimate, abs=1e-10)
    assert a.se == pytest.approx(b.se, abs=1e-10)
    assert a.simultaneous_ci == pytest.approx(b.simultaneous_ci)


def test_r_reference_comparison_refuses_unexecuted_fixtures(tmp_path):
    from validation.compare_phase6_r import compare
    with pytest.raises(FileNotFoundError, match="R producer outputs"):
        compare(tmp_path)
