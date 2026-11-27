"""Phase 11C: reproducible cross-fitting, independent OOF calculations, and limitations."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from numpy.testing import assert_allclose

import quasi
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.matching.psm import _stratified_folds


def data(n=850, seed=820):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = expit(-0.2 + 0.7 * x)
    d = rng.binomial(1, e)
    y = 0.4 + 0.7 * x + 1.75 * d + rng.normal(scale=0.9, size=n)
    return pd.DataFrame({"y": y, "d": d, "x": x})


def design(estimator="aipw", **kwargs):
    return quasi.ObservationalDesign(outcome="y", treatment="d", covariates=("x",),
                                     estimator=estimator, **kwargs)


def test_cross_fit_matches_independent_statsmodels_holdout_models():
    import statsmodels.api as sm
    df = data(1020)
    r = quasi.estimate(df, design=design(n_splits=5, fold_seed=91))
    d = df.d.to_numpy()
    y = df.y.to_numpy()
    z = sm.add_constant(df.x.to_numpy())
    folds = r.fold_assignments
    pred_e, pred_m = np.empty(len(df)), np.empty(len(df))
    for fold in range(5):
        train, held = folds != fold, folds == fold
        glm = sm.GLM(d[train], z[train], family=sm.families.Binomial()).fit()
        ols = sm.OLS(y[train & (d == 0)], z[train & (d == 0)]).fit()
        pred_e[held] = glm.predict(z[held])
        pred_m[held] = ols.predict(z[held])
    assert_allclose(r.out_of_fold_propensity, pred_e, rtol=2e-5, atol=1e-8)
    assert_allclose(r.out_of_fold_control_outcome, pred_m, rtol=2e-5, atol=1e-7)
    odds = (1-d) * pred_e/(1-pred_e)
    expected_tau = np.mean((d-odds)*(y-pred_m))/d.mean()
    influence = (d*(y-pred_m-expected_tau)-odds*(y-pred_m))/d.mean()
    expected_se = np.sqrt(np.mean((influence-influence.mean())**2)/len(d))
    assert r.estimate == pytest.approx(expected_tau, abs=2e-6)
    assert r.se == pytest.approx(expected_se, rel=2e-5)
    assert r.inference_details['method'] == 'cross_fitted_AIPW_ATT_EIF_HC0_experimental'
    assert r.inference_details['reference_certified'] is False
    assert r.identification_status == 'not_established'


def test_split_reproducibility_hash_and_heldout_provenance():
    df = data(620)
    dsg = design(n_splits=4, fold_seed=11)
    a = quasi.estimate(df, design=dsg)
    b = quasi.estimate(df, design=dsg)
    c = quasi.estimate(df, design=design(n_splits=4, fold_seed=12))
    assert_allclose(a.fold_assignments, b.fold_assignments)
    assert a.estimate == b.estimate
    assert np.any(a.fold_assignments != c.fold_assignments)
    assert np.array_equal(np.unique(a.fold_assignments), np.arange(4))
    meta = a.model_meta
    assert meta['nuisance_predictions'] == 'out_of_fold_only'
    assert meta['split_algorithm'] == 'stratified_numpy_PCG64_permutation_array_split_v1'
    assert meta['fold_assignment_sha256'] == hashlib.sha256(a.fold_assignments.astype('<i4').tobytes()).hexdigest()
    assert sum(row['heldout_n'] for row in meta['folds']) == len(df)
    assert sum(row['heldout_treated_n'] for row in meta['folds']) == df.d.sum()
    assert all(row['train_controls_n'] > 0 for row in meta['folds'])
    rec = quasi.ResultRecord.from_json(a.to_json())
    assert rec.inference.method == 'cross_fitted_AIPW_ATT_EIF_HC0_experimental'
    assert json.loads(a.to_json())['experimental'] is True
    assert 'not_established' == rec.identification_status


def test_holdout_outcomes_do_not_leak_into_own_nuisance_predictions():
    df = data(680)
    result = quasi.estimate(df, design=design(n_splits=3, fold_seed=9))
    first = result.fold_assignments == 0
    edited = df.copy()
    edited.loc[first, 'y'] += 15
    changed = quasi.estimate(edited, design=design(n_splits=3, fold_seed=9))
    assert_allclose(changed.out_of_fold_control_outcome[first], result.out_of_fold_control_outcome[first])
    assert_allclose(changed.out_of_fold_propensity[first], result.out_of_fold_propensity[first])
    assert np.any(np.abs(changed.out_of_fold_control_outcome[~first] - result.out_of_fold_control_outcome[~first]) > 0.1)


def test_crossfit_ipw_point_only_with_independent_hajek_formula():
    df = data(730)
    r = quasi.estimate(df, design=design('ipw', n_splits=4, fold_seed=5))
    d = df.d.to_numpy()
    y = df.y.to_numpy()
    e = r.out_of_fold_propensity
    w = (1-d) * e/(1-e)
    assert r.estimate == pytest.approx(y[d == 1].mean() - np.dot(w,y)/w.sum())
    assert np.isnan(r.se) and np.isnan(r.ci_lower) and np.isnan(r.p_value)
    assert r.inference_details['method'] == 'not_available'
    record = quasi.ResultRecord.from_json(r.to_json())
    assert record.inference.se is None
    assert record.inference.ci_lower is None
    assert r.model_meta['uncertainty'] == 'not_available_crossfit_IPW_nonorthogonal'
    assert 'point estimate only' in r.summary()


def test_defaults_unchanged_and_legacy_keyword_dispatch_available():
    df = data(650)
    original = quasi.AIPWATT(df, outcome='y', treatment='d', covariates=['x']).fit()
    unchanged = quasi.estimate(df, design=design(n_splits=1))
    assert original.estimate == pytest.approx(unchanged.estimate)
    assert original.se == pytest.approx(unchanged.se)
    r = quasi.estimate(df, outcome='y', method='aipw', treatment='d', covariates=('x',),
                       n_splits=3, fold_seed=8)
    assert r.model_meta['cross_fitted'] is True
    direct = quasi.AIPWATT(df, outcome='y', treatment='d', covariates=['x'], n_splits=3, fold_seed=8).fit()
    assert direct.estimate == r.estimate


@pytest.mark.parametrize('bad', [0, -2, 21, 2.5, True, '3'])
def test_invalid_crossfit_folds_rejected(bad):
    with pytest.raises(InvalidDesignError, match='n_splits'):
        design(n_splits=bad)


@pytest.mark.parametrize('bad', [-2, 2**32, True, 1.2, '7'])
def test_invalid_seed_rejected(bad):
    with pytest.raises(InvalidDesignError, match='fold_seed'):
        design(n_splits=3, fold_seed=bad)


def test_matching_cannot_crossfit():
    with pytest.raises(InvalidDesignError, match='not supported'):
        design('psm', n_splits=2)


def test_too_few_per_class_after_split_rejected():
    tiny = data(9, 2)
    tiny["d"] = [1]*4 + [0]*5
    with pytest.raises(InsufficientSupportError, match='(?:training|validation fold)'):
        quasi.estimate(tiny, design=design(n_splits=5))


def test_training_rank_failure_is_not_silently_regularized():
    df = data(250)
    # One control subgroup concentrated in one fold is not enough for stable
    # training. Infeasible splits must fail instead of truncating predictors.
    z = df[['x']].to_numpy()
    d = df.d.to_numpy()
    folds = _stratified_folds(d, n_splits=3, seed=19, min_training=4)
    # A near-constant predictor in one training fold causes a singular design.
    special = df.copy()
    special.loc[folds != 0, 'x'] = 1.5
    with pytest.raises((InvalidDesignError, InsufficientSupportError)):
        quasi.estimate(special, design=design(n_splits=3, fold_seed=19))
