"""SCM is NOT SDiD. Tests distinguish the objectives and uncertainty contracts."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import quasi
from quasi.errors import ConvergenceError, InvalidDesignError
from quasi.synth.scm import _fit_scm_weights


def panel(*, effect=2.0, noisy=False):
    rng = np.random.default_rng(100)
    periods = range(9)
    d1 = np.array([6 + .6 * t + .15 * np.sin(t) for t in periods])
    d2 = np.array([3 + .15 * t + .3 * np.cos(t) for t in periods])
    d3 = np.array([2 + .3 * t + .05 * t * t for t in periods])
    d4 = np.array([5 - .1 * t + .11 * np.sin(t * 2) for t in periods])
    # Exactly known synthetic combination, treatment begins period 6.
    treated = .3 * d1 + .7 * d2 + effect * (np.arange(9) >= 6)
    if noisy:
        treated += rng.normal(0, .03, len(treated))
    return pd.DataFrame([(name, t, float(y)) for name, array in (
        ('treated', treated), ('don1', d1), ('don2', d2), ('don3', d3), ('don4', d4))
        for t, y in enumerate(array)], columns=['unit', 'time', 'y'])


def test_classic_scm_recovers_hand_known_convex_trajectory():
    r = quasi.SyntheticControl(panel(), 'y', 'unit', 'time', 'treated', 6, n_placebos=0).fit()
    assert r.estimate == pytest.approx(2., abs=3e-4)
    assert r.pre_rmspe < 3e-4
    assert sum(r.donor_weights.values()) == pytest.approx(1.)
    assert all(0 <= w <= 1 for w in r.donor_weights.values())
    assert np.allclose(r.synthetic_trajectory[:6], r.actual_trajectory[:6], atol=3e-4)
    assert np.isnan(r.se) and np.isnan(r.ci_lower)
    assert 'not available' in r.summary().lower()
    assert 'not_available' == r.inference_details['method']
    assert r.placebo_p_value is None
    assert r.model_name != quasi.SyntheticDiD(panel(), 'y', 'unit', 'time', 'treated', 6, n_placebos=0).fit().model_name


def test_public_explicit_spec_and_legacy_selector_agree():
    data = panel(noisy=True)
    design = quasi.SyntheticDesign(outcome='y', unit='unit', time='time',
        treated_unit='treated', intervention_time=6, estimator='scm', n_placebos=3, seed=9)
    a = quasi.estimate(data, design=design)
    b = quasi.estimate(data, 'y', method='scm', unit='unit', time='time',
        treated_unit='treated', intervention_time=6, n_placebos=3, seed=9)
    assert a.estimate == pytest.approx(b.estimate)
    assert a.model_name == 'SyntheticControl'
    assert a._design_spec.family == 'scm'
    assert a.placebo_p_value == b.placebo_p_value
    assert a.to_record().inference.se is None
    assert a.to_metadata_dict()['primary_metric']['se'] is None
    encoded = a.to_json()
    assert 'NaN' not in encoded
    assert json.loads(encoded)['inference']['method'] == 'not_available'
    assert quasi.ResultRecord.from_json(encoded).to_json() == encoded
    assert a.to_record().provenance.method_spec_json
    assert 'Synthetic Control' in a.methods_text()
    assert all(a.donor_weights[u] >= 0 for u in a.donor_weights)


def test_sdid_default_and_no_alias_legacy_behavior():
    df = panel(effect=1)
    spec = quasi.SyntheticDesign(outcome='y', unit='unit', time='time',
        treated_unit='treated', intervention_time=6, n_placebos=0)
    a = quasi.estimate(df, design=spec)
    assert a.model_name == 'SyntheticDiD'
    assert spec.family == 'sdid'
    assert quasi.SyntheticControl is not quasi.SyntheticDiD
    assert quasi.SyntheticControl.__mro__[1] is quasi.BaseQEDModel


def test_scm_checked_optimum_matches_separate_slsqp_formulation():
    rng = np.random.default_rng(12)
    ctrl = rng.normal(size=(7, 15))
    true_w = np.array([.2, 0, .13, .1, .37, .2, 0.])
    target = true_w @ ctrl + rng.normal(scale=.01, size=15)
    w = _fit_scm_weights(target, ctrl)
    result = minimize(lambda z: np.sum((z @ ctrl - target) ** 2),
                      np.ones(7)/7, method='SLSQP', bounds=[(0,1)]*7,
                      constraints=[{'type':'eq', 'fun':lambda z: z.sum() - 1}],
                      options={'ftol':1e-12, 'maxiter':3000})
    assert result.success
    assert np.sum((w @ ctrl - target)**2) == pytest.approx(result.fun, abs=1e-7)


def test_placebo_pool_excludes_actual_treated_and_deterministic(monkeypatch):
    from quasi.synth import scm
    df = panel(noisy=True)
    original = scm._fit_scm_weights
    calls = []
    def watch(target, donors):
        calls.append((target.copy(), donors.shape[0]))
        return original(target, donors)
    monkeypatch.setattr(scm, '_fit_scm_weights', watch)
    a = quasi.SyntheticControl(df, 'y', 'unit', 'time', 'treated', 6, n_placebos=3, seed=17).fit()
    b = quasi.SyntheticControl(df, 'y', 'unit', 'time', 'treated', 6, n_placebos=3, seed=17).fit()
    assert np.array_equal(a.placebo_ratios, b.placebo_ratios)
    assert calls[0][1] == 4
    assert all(n == 3 for _, n in calls[1:4])
    assert a.model_meta['placebo_count'] == len(a.placebo_ratios)
    if a.placebo_p_value is not None:
        assert a.placebo_p_value >= 1/(len(a.placebo_ratios)+1)


@pytest.mark.parametrize('mutation', [
    lambda d: d.drop(d.index[0]),
    lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True),
    lambda d: d.assign(y=np.inf),
    lambda d: d.assign(unit=lambda x: x.unit.where(x.unit != 'don1', 1)),
    lambda d: d.assign(y=lambda x: x.y.where(x.time != 2, np.nan)),
])
def test_bad_panel_fails_closed(mutation):
    with pytest.raises((InvalidDesignError, ValueError, TypeError)):
        quasi.SyntheticControl(mutation(panel()), 'y', 'unit', 'time', 'treated', 6).fit()


def test_scm_rejects_missing_intervention_and_unsupported_covariates():
    with pytest.raises((InvalidDesignError, ValueError), match='Intervention'):
        quasi.SyntheticControl(panel(), 'y', 'unit', 'time', 'treated', 99).fit()
    with pytest.raises(NotImplementedError):
        quasi.SyntheticControl(panel(), 'y', 'unit', 'time', 'treated', 6, covariates=['age'])
    with pytest.raises(InvalidDesignError):
        quasi.SyntheticDesign(outcome='y', unit='unit', time='time',
            treated_unit='treated', intervention_time=6, estimator='not_scm')
    with pytest.raises(InvalidDesignError):
        quasi.estimate(panel(), 'y', method='scm', unit='unit', time='time',
            treated_unit='treated', intervention_time=6, estimator='sdid')


def test_not_available_inference_only_with_all_null_fields():
    from quasi.results.schema import InferenceRecord
    with pytest.raises(ValueError):
        InferenceRecord(method='not_available', confidence_level=.95,
            se=None, p_value=0, ci_lower=None, ci_upper=None)
    with pytest.raises(ValueError):
        InferenceRecord(method='HC0', confidence_level=.95,
            se=None, p_value=None, ci_lower=None, ci_upper=None)
    r = quasi.estimate(panel(), 'y', method='scm', unit='unit', time='time',
                       treated_unit='treated', intervention_time=6, n_placebos=0)
    payload = json.loads(r.to_json())
    payload['inference']['p_value'] = 0.01
    with pytest.raises(ValueError):
        quasi.ResultRecord.from_json(json.dumps(payload))


def test_leave_one_donor_out_refits_pre_only_and_is_descriptive():
    r = quasi.SyntheticControl(panel(noisy=True), 'y', 'unit', 'time', 'treated', 6,
                               n_placebos=0).fit()
    table = r.leave_one_donor_out()
    assert len(table) == 4
    assert set(table.excluded_donor) == set(r.donor_weights)
    np.testing.assert_allclose(table.post_average_gap - r.estimate, table.difference_from_full)
    assert r.model_meta['donor_herfindahl'] == pytest.approx(sum(w*w for w in r.donor_weights.values()))
    assert 1 <= r.model_meta['effective_donors'] <= 4


def test_legacy_classic_optimizer_uses_checked_public_objective():
    from quasi.synth import ClassicSCM
    df = panel()
    pivot = df.pivot(index='unit', columns='time', values='y')
    ids = ['don1','don2','don3','don4']
    w = ClassicSCM(pivot.loc['treated'].values[:6], pivot.loc[ids].values[:, :6]).fit_weights()
    assert w.sum() == pytest.approx(1)
    assert np.linalg.norm(w @ pivot.loc[ids].values[:, :6] - pivot.loc['treated'].values[:6]) < 1e-3


def test_scm_donor_weights_do_not_leak_posttreatment_outcomes():
    df = panel(noisy=True)
    a = quasi.SyntheticControl(df, 'y', 'unit','time','treated', 6, n_placebos=0).fit()
    changed = df.copy()
    changed.loc[(changed.unit == 'don1') & (changed.time >= 6), 'y'] += 1000
    b = quasi.SyntheticControl(changed, 'y', 'unit','time','treated', 6, n_placebos=0).fit()
    assert a.donor_weights == b.donor_weights
    assert a.pre_rmspe == pytest.approx(b.pre_rmspe)
    # Counterfactual *is* sensitive to donor exposure after treatment.
    assert abs(a.estimate - b.estimate) > 1.
