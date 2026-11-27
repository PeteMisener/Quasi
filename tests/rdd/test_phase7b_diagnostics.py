"""Phase 7B safeguards; statistical reference parity is a separate open gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import quasi

from quasi.base import DiagnosticStatus
from quasi.rdd.diagnostics import covariate_continuity
from quasi.diagnostics.density import mccrary_density_test


def data(seed=32, n=650):
    g = np.random.default_rng(seed)
    x = g.uniform(-2, 2, size=n)
    x[:8] = 0.0
    e = g.normal(size=n)
    z1 = 1.3 * (x >= 0) + 0.3 * x + g.normal(0, .3, n)
    z2 = .5 * x + g.normal(0, 1, n)
    d = (g.random(n) < np.where(x >= 0, .8, .2)).astype(float)
    y = 1.4 * (x >= 0) + .45 * x + .25 * x * x + e
    return pd.DataFrame(dict(x=x, y=y, z1=z1, z2=z2, d=d))


def model(df, **kw):
    return quasi.SharpRDD(df, outcome='y', running_var='x', cutoff=0.,
                          bandwidth=1.4, **kw)


def test_donut_same_as_manual_filtered_regression_for_both_point_and_se():
    df = data()
    radius = .21
    full = model(df).fit()
    donut = model(df, donut_radius=radius).fit()
    manual = model(df.loc[df.x.abs() > radius]).fit()
    assert donut.estimate == pytest.approx(manual.estimate, abs=1e-11)
    assert donut.se == pytest.approx(manual.se, abs=1e-11)
    assert donut.inference_details['n_donut_excluded'] == (df.x.abs() <= radius).sum()
    assert donut.n_obs == len(df)
    assert donut.n_effective < full.n_effective
    assert donut.inference_details['donut_radius'] == radius
    assert donut.diagnostics['Donut Extrapolation'].status == DiagnosticStatus.WARN
    assert 'extrapolat' in donut.methods_text()


def test_donut_preserves_baseline_data_and_has_explicit_zero_case():
    df = data()
    before = df.copy(deep=True)
    fitted = model(df).fit()
    sens = fitted.donut_sensitivity(radii=[0., .12, .22])
    assert sens.rows[0].estimate == pytest.approx(fitted.estimate)
    assert sens.rows[0].se == pytest.approx(fitted.se)
    assert sens.rows[0].n_excluded == 0
    assert [row.n_excluded for row in sens.rows] == sorted(row.n_excluded for row in sens.rows)
    pd.testing.assert_frame_equal(df, before)
    assert fitted.inference_details['donut_radius'] == 0
    assert 'Donut Extrapolation' not in fitted.diagnostics
    assert set(['donut_radius', 'n_excluded']).issubset(sens.to_dataframe())


def test_donut_fuzzy_and_bias_corrected_fit_on_identical_selected_rows():
    df = data()
    donut_df = df[df.x.abs() > .12]
    kwargs = dict(outcome='y', running_var='x', cutoff=0., bandwidth=1.4,
                  inference='bias_corrected', bias_bandwidth=1.55)
    for cls, extra in [(quasi.SharpRDD, {}), (quasi.FuzzyRDD, {'treatment': 'd'})]:
        left = cls(df, **kwargs, donut_radius=.12, **extra).fit()
        right = cls(donut_df, **kwargs, **extra).fit()
        assert left.estimate == pytest.approx(right.estimate, rel=1e-11)
        assert left.se == pytest.approx(right.se, rel=1e-11)


def test_design_router_explicit_and_legacy_donut_match_direct_fit():
    df = data()
    spec = quasi.RDDDesign(outcome='y', running_var='x', cutoff=0., bandwidth=1.4,
                           donut_radius=.17, diagnostic_covariates=('z1', 'z2'))
    from_design = quasi.estimate(df, design=spec)
    direct = model(df, donut_radius=.17, diagnostic_covariates=['z1', 'z2']).fit()
    legacy = quasi.estimate(df, outcome='y', running_var='x', cutoff=0., bandwidth=1.4,
                            donut_radius=.17, diagnostic_covariates=['z1', 'z2'])
    assert from_design.estimate == pytest.approx(direct.estimate)
    assert legacy.se == pytest.approx(direct.se)
    assert from_design.inference_details['diagnostic_covariates'] == ['z1', 'z2']
    assert 'Covariate continuity: z1' in from_design.diagnostics
    assert from_design.diagnostics['Covariate continuity: z1'].status == DiagnosticStatus.WARN


def test_predetermined_covariates_do_not_adjust_treatment_estimate():
    df = data()
    a = model(df).fit()
    b = model(df, diagnostic_covariates=['z1','z2']).fit()
    assert a.estimate == pytest.approx(b.estimate, abs=1e-12)
    assert a.se == pytest.approx(b.se, abs=1e-12)
    assert b.diagnostics['Covariate continuity: z1'].p_value < .05
    assert b.diagnostics['Covariate continuity: z2'].p_value > .05
    assert b.identification_status == 'not_established'


def test_all_invalid_donut_radii_fail_closed():
    df = data()
    for radius in [-1, np.nan, np.inf, True, 1.4, 2.0]:
        with pytest.raises(ValueError, match='donut_radius'):
            model(df, donut_radius=radius)
        with pytest.raises(ValueError, match='donut_radius'):
            quasi.RDDDesign(outcome='y', running_var='x', cutoff=0,
                            bandwidth=1.4, donut_radius=radius)
    r = model(df).fit()
    for bad in [np.nan, -0.1, 1.4, True]:
        with pytest.raises(ValueError, match='Donut radius'):
            r.donut_sensitivity([bad])


def test_explicit_preflight_rejects_donut_insufficient_support():
    df = data(n=300)
    spec = quasi.RDDDesign(outcome='y', running_var='x', cutoff=0, bandwidth=.4, donut_radius=.39)
    with pytest.raises(ValueError, match='support'):
        quasi.estimate(df, design=spec)


def test_donut_sensitivity_small_support_error_names_radius():
    df = data(n=300)
    result = model(df).fit()
    with pytest.raises(ValueError, match='Unsupported donut radius'):
        result.donut_sensitivity([1.39])


def test_invalid_diagnostic_names_and_missing_values_raise():
    df = data()
    for cols in [['x'], ['y'], ['z1', 'z1'], [''], ['z2','d','y']]:
        with pytest.raises(ValueError, match='[Dd]iagnostic'):
            model(df, diagnostic_covariates=cols)
    with pytest.raises(ValueError):
        quasi.RDDDesign(outcome='y', running_var='x', cutoff=0,
                        diagnostic_covariates=('z1', 'z1'))
    df.loc[0, 'z1'] = np.inf
    with pytest.raises(ValueError, match='[Dd]iagnostic|finite'):
        model(df, diagnostic_covariates=['z1'])


def test_constant_covariate_flagged_as_unavailable_not_considered_balance():
    df = data()
    df['constant'] = 1.0
    result = model(df, diagnostic_covariates=['constant']).fit()
    diag = result.diagnostics['Covariate continuity: constant']
    assert diag.status == DiagnosticStatus.WARN
    assert diag.p_value is None
    assert 'no continuity test' in diag.message


def test_holm_p_adjustment_matches_manual_monotone_rule(monkeypatch):
    from quasi.rdd import diagnostics
    zvalues = [2.5758293035489, 2.0537489106318, 1.8807936081513]
    seq = iter(zvalues)
    monkeypatch.setattr(diagnostics, 'rd_inference', lambda *args, **kwargs: {
        'estimate': next(seq), 'se': 1.0})
    x = np.linspace(-1, 1, 30)
    r = covariate_continuity(x, {'a':x, 'b':x, 'c':x}, 1.0, 'uniform', 1)
    assert [r[f'Covariate continuity: {key}'].p_value for key in 'abc'] == pytest.approx(
        [.03, .08, .08], abs=.0001)


def test_mass_points_report_counts_and_warn_despite_estimable_fit():
    rng = np.random.default_rng(5)
    x = rng.choice(np.linspace(-1.9,1.9,33), size=800)
    y = 1 + (x>=0)*2 + .2*x + rng.normal(0,.3,len(x))
    df = pd.DataFrame({'x':x,'y':y})
    result = model(df).fit()
    support = result.diagnostics['Running Variable Support']
    assert support.status == DiagnosticStatus.WARN
    assert 'unique scores' in support.message
    assert result.inference_details['unique_left'] >= 3
    assert result.inference_details['unique_right'] >= 3
    assert not result.passed_validation


def test_exploratory_density_screen_cannot_assert_definitive_failure():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(-1,0, 300), rng.uniform(0,.25,600)])
    d = mccrary_density_test(x, 0.)
    assert d.status == DiagnosticStatus.WARN
    assert 'exploratory' in d.name.lower()
    assert 'uncalibrated' in d.threshold


def test_robustness_grid_validates_options_and_labels_requested_confidence_level():
    df = data()
    result = model(df, alpha=.10).fit()
    grid = result.robustness(bandwidth_multipliers=[1.0], kernels=['triangular'],orders=[1])
    assert grid.confidence_level == pytest.approx(.9)
    assert grid.rows[0].estimate == pytest.approx(result.estimate)
    assert grid.rows[0].n_excluded == 0
    for bad in [0., -1., np.inf, np.nan, True]:
        with pytest.raises(ValueError, match='multipliers'):
            result.robustness(bandwidth_multipliers=[bad])
    with pytest.raises(ValueError, match='kernel'):
        result.robustness(kernels=['nope'])
    with pytest.raises(ValueError, match='orders'):
        result.robustness(orders=[3])


def test_robustness_grid_reflects_baseline_donut_exclusion():
    df = data()
    result = model(df, donut_radius=.16).fit()
    grid = result.robustness(bandwidth_multipliers=[1.0],kernels=['triangular'],orders=[1])
    assert grid.rows[0].estimate == pytest.approx(result.estimate)
    assert grid.rows[0].donut_radius == .16
    assert grid.rows[0].n_excluded == result.inference_details['n_donut_excluded']


def test_heuristic_bandwidth_pilot_refuses_rank_deficient_discrete_running_score():
    rng = np.random.default_rng(29)
    x = rng.choice([-.5, -.25, .25, .5], size=800)
    df = pd.DataFrame({'x':x, 'y': 1 + 2*(x>=0) + rng.normal(size=len(x))})
    with pytest.raises(ValueError, match='bandwidth pilot lacks three distinct'):
        quasi.SharpRDD(df, 'y', 'x', 0, bandwidth='cct_optimal').fit()


def test_diagnostic_covariates_string_not_treated_as_character_names():
    with pytest.raises(ValueError, match='diagnostic_covariates'):
        model(data(), diagnostic_covariates='z1')
    with pytest.raises(ValueError, match='diagnostic_covariates'):
        quasi.RDDDesign(outcome='y', running_var='x', cutoff=0,
                        diagnostic_covariates='z1')
