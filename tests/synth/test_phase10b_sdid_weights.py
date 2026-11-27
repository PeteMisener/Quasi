"""Independent optimization and evidence-contract tests, not external R parity."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import quasi
from quasi.synth.sdid import _simplex_regression, _sdid_weights, _sdid_effect
from quasi.errors import InvalidDesignError, ConvergenceError


def reference_slsqp(A, b, ridge, intercept=True):
    """Independent objective with intercept estimated directly, not demeaned."""
    m, n = A.shape
    def objective(v):
        w, a = v[:n], v[n] if intercept else 0.
        r = b - A @ w - a
        return (r @ r) / m + ridge**2 * (w @ w)
    x = np.r_[np.ones(n) / n, 0.]
    fit = minimize(objective, x, method='SLSQP',
                   bounds=[(0, 1)]*n+[(None,None)],
                   constraints={'type': 'eq', 'fun': lambda v: v[:n].sum()-1},
                   options={'ftol': 1e-12, 'maxiter': 3000})
    assert fit.success
    return fit.x[:n], objective(fit.x)


def dataset(seed=81, n=8, t=14, onset=9, effect=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(2,t))
    weights = rng.dirichlet(np.ones(n))
    donors = (rng.normal(size=(n,2)) @ f
              + rng.normal(size=(n,1))*2
              + np.linspace(0,3,t)[None,:]
              + rng.normal(scale=.08,size=(n,t)))
    treated = weights @ donors + 4 + effect*(np.arange(t)>=onset)
    rows = [('treated', i, float(y)) for i,y in enumerate(treated)]
    rows += [(f'd{j}',i,float(y)) for j,track in enumerate(donors) for i,y in enumerate(track)]
    return pd.DataFrame(rows,columns=['unit','time','y']), treated, donors


def test_intercept_simplex_matches_independently_optimized_objective():
    rng = np.random.default_rng(88)
    A = rng.normal(size=(30,5)); b = A @ np.array([.1,.0,.6,.2,.1])+7+rng.normal(0,.03,30)
    for ridge in [0, 0.025, .5]:
        w, a = _simplex_regression(A, b, ridge)
        other, optimal_obj = reference_slsqp(A, b, ridge)
        got_obj = np.mean((b - A @ w - a)**2) + ridge**2 * (w@w)
        assert got_obj == pytest.approx(optimal_obj, abs=1e-8)
        assert np.all(w >= 0) and np.isclose(w.sum(),1)


def test_unit_and_time_weights_have_reference_regularization_and_intercepts():
    _, t, donors = dataset()
    omega, lam, details = _sdid_weights(t, donors,9)
    expected_sigma = np.std(np.diff(donors[:,:9],axis=1),ddof=1)
    assert details['noise_sd_first_differences'] == pytest.approx(expected_sigma)
    assert details['zeta_omega'] == pytest.approx(expected_sigma*5**.25)
    assert details['zeta_lambda'] == pytest.approx(expected_sigma*1e-6)
    assert omega.sum() == pytest.approx(1.)
    assert lam.sum() == pytest.approx(1.)
    assert np.all(omega >= 0) and np.all(lam >= 0)
    assert details['sparsification'] == 'none'


def test_unit_weight_intercept_invariant_to_treated_level_shift():
    _, t, donors = dataset()
    omega, lam, d = _sdid_weights(t,donors,9)
    w2, l2, d2 = _sdid_weights(t+19,donors,9)
    np.testing.assert_allclose(omega,w2,atol=1e-6)
    np.testing.assert_allclose(lam,l2,atol=1e-6)
    assert d2['omega_intercept'] == pytest.approx(d['omega_intercept']+19)


def test_time_weight_intercept_invariant_to_common_donor_post_shift():
    _, t, donors = dataset()
    _, lam, d = _sdid_weights(t, donors,9)
    shifted = donors.copy(); shifted[:,9:] += 55
    _, l2, d2 = _sdid_weights(t, shifted, 9)
    np.testing.assert_allclose(lam,l2,atol=1e-6)
    assert d2['lambda_intercept'] == pytest.approx(d['lambda_intercept']+55)


def test_point_effect_matches_matrix_bilinear_form():
    _, t, donors = dataset()
    omega, lam, _ = _sdid_weights(t, donors, 9)
    combined_y = np.vstack([donors, t])
    row_weights = np.r_[-omega, 1.]
    time_weights = np.r_[-lam,np.ones(5)/5]
    assert _sdid_effect(t, donors, 9,omega,lam) == pytest.approx(row_weights@combined_y@time_weights)


def test_public_result_has_explicitly_missing_uncertainty_and_valid_json():
    frame, _, _ = dataset()
    r = quasi.SyntheticDiD(frame,'y','unit','time','treated',9,n_placebos=2).fit()
    assert r.model_name == 'SyntheticDiD'
    assert np.isfinite(r.estimate) and np.isnan(r.se)
    assert r.inference_details['method'] == 'not_available'
    assert r.to_metadata_dict()['primary_metric']['ci'] == (None,None)
    record = json.loads(r.to_json())
    assert record['inference']['method'] == 'not_available'
    assert all(v is None for v in [record['inference']['se'], record['inference']['p_value']])
    assert len(r.donor_weights) == 8 and len(r.time_weights) == 9
    assert r.pre_rmspe <= r.raw_pre_rmspe + 1e-10


def test_no_donor_contamination_in_placebos(monkeypatch):
    frame, actual, donors = dataset()
    from quasi.synth import sdid
    inputs=[]
    original=sdid._sdid_weights
    def traced(t,d,k,**kw):
        inputs.append((t.copy(),d.copy()))
        return original(t,d,k,**kw)
    monkeypatch.setattr(sdid,'_sdid_weights',traced)
    r=quasi.SyntheticDiD(frame,'y','unit','time','treated',9,n_placebos=4,seed=17).fit()
    # Full-sample first, then eight leave-one-donor-out, then placebos.
    for pseudo_target, pool in inputs[9:]:
        assert not any(np.array_equal(row,actual) for row in pool)
        assert not any(np.array_equal(row,pseudo_target) for row in pool)
    assert r.model_meta['placebos_attempted'] == 4
    assert r.model_meta['placebos_succeeded'] == len(r.placebo_effects)


def test_donor_exclusion_validation_fails_closed():
    frame,_,_=dataset()
    frame=frame.loc[~((frame.unit=='d0')&(frame.time==2))]
    with pytest.raises(ValueError,match='balanced|complete'):
        quasi.SyntheticDiD(frame,'y','unit','time','treated',9).fit()


def test_invalid_regularization_rejected():
    _,t,d=dataset()
    with pytest.raises(InvalidDesignError):
        _sdid_weights(t,d,9,zeta_omega=-1)


def test_seeded_placebos_are_deterministic():
    frame,_,_=dataset()
    a=quasi.SyntheticDiD(frame,'y','unit','time','treated',9,n_placebos=3,seed=8).fit()
    b=quasi.SyntheticDiD(frame,'y','unit','time','treated',9,n_placebos=3,seed=8).fit()
    np.testing.assert_allclose(a.placebo_effects,b.placebo_effects)
    assert a.placebo_p_val == b.placebo_p_val


def test_clean_additive_effect_recovery_with_level_shift():
    rng=np.random.default_rng(331)
    time=np.arange(14)
    common=.3*time+np.sin(time/3)
    donor=np.array([common+v for v in [-2.,0.,3.,5.]])
    actual=common+9.+4*(time>=9)
    rows=[('treated',i,float(y)) for i,y in enumerate(actual)]
    rows += [(f'd{j}',i,float(y)) for j,track in enumerate(donor) for i,y in enumerate(track)]
    r=quasi.SyntheticDiD(pd.DataFrame(rows,columns=['unit','time','y']),
                         'y','unit','time','treated',9,n_placebos=0).fit()
    assert r.estimate == pytest.approx(4.,abs=1e-7)
    assert r.inference_details['reference_certified'] is False
