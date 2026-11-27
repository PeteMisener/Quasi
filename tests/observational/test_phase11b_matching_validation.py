"""Iteration 23: reproducible matching, empirical overlap, and honest inference."""
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import quasi
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.matching.psm import _nearest_controls, _overlap_summary, _propensity


def data(seed=231, n=300):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = rng.binomial(1, 1/(1+np.exp(-(0.2 + x*0.7))))
    y = 1 + 0.5*x + 2*d + rng.normal(size=n)
    return pd.DataFrame({'y': y, 'd': d, 'x': x})


def test_k_nearest_exact_match_against_independent_full_sort():
    df = data()
    e = _propensity(np.c_[np.ones(len(df)), df.x], df.d.to_numpy())
    ti = np.where(df.d.to_numpy()==1)[0]
    ci = np.where(df.d.to_numpy()==0)[0]
    for k in (1, 2, 3, 7):
        r = quasi.PropensityScoreMatching(df, outcome='y', treatment='d', covariates=['x'], n_neighbors=k).fit()
        expected = np.stack([ci[np.lexsort((ci, np.abs(e[ci]-e[t])))[:k]] for t in ti])
        observed = r.matched_control_indices[:,None] if k == 1 else r.matched_control_indices
        assert_allclose(observed, expected)
        assert r.estimate == pytest.approx(np.mean(df.y.to_numpy()[ti] - df.y.to_numpy()[expected].mean(axis=1)))
        assert all(len(set(row)) == k for row in observed)
        assert not np.isfinite(r.se)
        assert json.loads(r.to_json())['inference']['se'] is None
        assert r.model_meta['max_control_reuse_count'] >= 1
        assert r.model_meta['n_neighbors'] == k
        assert r.model_meta['effective_matched_control_n'] <= r.model_meta['unique_matched_controls'] + 1e-10
        assert r.matched_control_weights.sum() == pytest.approx(len(ti))


def test_exact_ties_original_index_order_and_unique_per_set():
    scores = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.9])
    ti = np.array([5])
    # Intentionally permute candidate list; tie rule uses source position.
    ci = np.array([3, 0, 4, 2, 1])
    matched, gaps = _nearest_controls(scores, ti, ci, 3)
    assert matched.tolist() == [[0, 1, 2]]
    assert_allclose(gaps, np.full((1,3), .5))
    assert len(np.unique(matched)) == 3


def test_large_matching_uses_per_treated_distance_vector_not_pairwise_matrix():
    rng = np.random.default_rng(42)
    scores = rng.uniform(size=9000)
    treated = np.arange(6500, 9000)
    control = np.arange(6500)
    matched, gaps = _nearest_controls(scores, treated, control, 2)
    assert matched.shape == (2500,2)
    assert np.isfinite(gaps).all()
    assert np.all(gaps[:,0] <= gaps[:,1])


def test_neighbor_count_and_caliper_are_fail_closed():
    df = data()
    for bad in [0, -1, True, False, 2., '3', None]:
        with pytest.raises(InvalidDesignError, match='n_neighbors'):
            quasi.ObservationalDesign(outcome='y',treatment='d',covariates=('x',), estimator='psm', n_neighbors=bad)
    with pytest.raises(InvalidDesignError, match='n_neighbors'):
        quasi.ObservationalDesign(outcome='y',treatment='d',covariates=('x',), estimator='aipw',n_neighbors=2)
    with pytest.raises(InsufficientSupportError, match='n_neighbors'):
        quasi.PropensityScoreMatching(df,'y','d',['x'], n_neighbors=len(df)).fit()
    with pytest.raises(InsufficientSupportError, match='Caliper'):
        quasi.PropensityScoreMatching(df,'y','d',['x'],n_neighbors=2,caliper=1e-12).fit()


def test_explicit_design_and_legacy_route_propagate_neighbor_count():
    df=data()
    design = quasi.ObservationalDesign(outcome='y',treatment='d',covariates=('x',),estimator='psm',n_neighbors=3)
    a = quasi.estimate(df, design=design)
    b = quasi.estimate(df, outcome='y', treatment='d', covariates=('x',), method='psm', n_neighbors=3)
    assert a.estimate == pytest.approx(b.estimate)
    assert a.matched_control_indices.shape[1] == 3
    assert json.loads(a.to_json())['additional_metadata']['model_meta']['n_neighbors'] == 3


def test_empirical_score_support_is_descriptive_no_certification():
    d = np.array([1.,1.,1.,0.,0.,0.])
    e = np.array([.1,.4,.8,.2,.3,.5])
    a = _overlap_summary(e,d)
    assert a['fraction_treated_outside_control_score_range'] == pytest.approx(2/3)
    assert a['fraction_control_outside_treated_score_range'] == 0
    assert a['empirical_overlap_range_width'] == pytest.approx(.3)
    r = quasi.AIPWATT(data(),'y','d',['x']).fit()
    assert r.model_meta['fraction_treated_outside_control_score_range'] >= 0
    assert r.identification_status == 'not_established'
    assert 'Empirical propensity support' in r.diagnostics


def test_real_data_fixture_numeric_reference():
    from validation.observational.reference_phase11b import verify_reference
    report = verify_reference()
    assert report['passed']
    assert report['reference_source'] == 'statsmodels_glm_and_control_ols'
    assert report['causal_claim'] is False
    assert len(report['comparisons']) == 11


def test_real_data_reference_rejects_tampered_fixture(tmp_path, monkeypatch):
    from validation.observational import reference_phase11b as ref
    source = ref._source_csv_bytes
    monkeypatch.setattr(ref, '_source_csv_bytes', lambda: source() + b'\n')
    with pytest.raises(AssertionError, match='content has changed'):
        ref.verify_reference()


def test_real_data_reference_rejects_modified_claims(tmp_path, monkeypatch):
    from validation.observational import reference_phase11b as ref
    payload = json.loads(ref.REFERENCE.read_text())
    payload['covariates'] = ['age']
    bad = tmp_path / 'reference.json'
    bad.write_text(json.dumps(payload))
    monkeypatch.setattr(ref, 'REFERENCE', bad)
    with pytest.raises(AssertionError, match='specification mismatch'):
        ref.verify_reference()


def test_real_data_numeric_glm_propensities_agree_with_unpenalized_mle():
    from validation.observational.reference_phase11b import _source_frame, FEATURES
    import statsmodels.api as sm
    from scipy.special import expit
    df=_source_frame()
    z=np.c_[np.ones(len(df)),df[list(FEATURES)].to_numpy()]
    d=df.treatment.to_numpy()
    external=sm.GLM(d,z,family=sm.families.Binomial()).fit()
    assert_allclose(_propensity(z,d),expit(z@external.params),atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('method', ['aipw', 'ipw', 'psm'])
def test_no_causal_identification_from_numerical_adequacy(method):
    r=quasi.estimate(data(), design=quasi.ObservationalDesign(
        outcome='y',treatment='d',covariates=('x',),estimator=method))
    assert r.identification_status == 'not_established'
    assert r.inference_details['reference_certified'] is False
    assert 'not_available' == r.inference_details['method'] if method == 'psm' else True
