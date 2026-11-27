"""Phase 11A: observational estimand and inference contracts, not causal certification."""
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

import quasi
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.matching.psm import _balance, _propensity


def sample(n=900, seed=141):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = expit(0.15 + 0.8 * x)
    d = rng.binomial(1, e)
    y = 1 + x + 2.0 * d + rng.normal(scale=0.6, size=n)
    return pd.DataFrame(dict(y=y, d=d, x=x))


def design(method='aipw', **kwargs):
    return quasi.ObservationalDesign(outcome='y', treatment='d', covariates=('x',), estimator=method, **kwargs)


def test_explicit_aipw_point_estimate_and_serialization():
    r = quasi.estimate(sample(), design=design())
    assert r.model_name == 'AIPWATT'
    assert abs(r.estimate - 2) < 0.15
    assert r.se > 0
    assert r.inference_details['reference_certified'] is False
    record = quasi.ResultRecord.from_json(r.to_json())
    assert record.inference.se == pytest.approx(r.se)
    assert record.inference.method == 'stacked_logit_control_OLS_ATT_M_estimator_HC0'
    assert record.estimand.endswith('(ATT)')
    assert record.identification_status == 'not_established'


def test_pure_matching_point_estimate_is_different_and_inference_is_missing():
    df = sample()
    aipw = quasi.estimate(df, design=design())
    r = quasi.estimate(df, design=design('psm'))
    assert r.model_name == 'PropensityScoreMatching'
    assert abs(r.estimate - aipw.estimate) > 1e-8
    ti = r.matched_treated_indices
    ci = r.matched_control_indices
    assert len(ti) == int(df.d.sum())
    assert np.all(df.d.to_numpy()[ci] == 0)
    assert r.estimate == pytest.approx(np.mean(df.y.to_numpy()[ti] - df.y.to_numpy()[ci]))
    rec = quasi.ResultRecord.from_json(r.to_json())
    assert rec.inference.method == 'not_available'
    assert rec.inference.se is None and rec.inference.ci_lower is None
    assert json.loads(r.to_json())['experimental'] is True
    assert 'unavailable' in r.summary()


def test_legacy_aipw_method_explicit_and_direct_constructor():
    data = sample()
    r1 = quasi.estimate(data, outcome='y', method='aipw', treatment='d', covariates=('x',))
    r2 = quasi.AIPWATT(data, outcome='y', treatment='d', covariates=['x']).fit()
    assert r1.estimate == pytest.approx(r2.estimate)
    r3 = quasi.estimate(data, outcome='y', method='psm', treatment='d', covariates=('x',))
    assert r3.model_name == 'PropensityScoreMatching'


def test_caliper_blocks_population_changes():
    data = sample(700)
    with pytest.raises(InsufficientSupportError, match='Caliper'):
        quasi.estimate(data, design=design('psm', caliper=1e-12))
    with pytest.raises(InvalidDesignError):
        design('aipw', caliper=0.2)
    with pytest.raises(InvalidDesignError):
        design('psm', caliper=-0.1)
    with pytest.raises(InvalidDesignError):
        design('ipw', caliper=0.05)
    with pytest.raises(InvalidDesignError):
        design('psm', caliper="tight")
    with pytest.raises(InvalidDesignError):
        design('invalid')
    with pytest.raises(InvalidDesignError):
        quasi.estimate(data, method='aipw', outcome='y', treatment='d', covariates=('x',), estimator='psm')


def test_outcome_model_rank_deficiency_is_rejected():
    df = sample()
    df['x2'] = df.x
    with pytest.raises(InvalidDesignError, match='rank-deficient'):
        quasi.estimate(df, design=quasi.ObservationalDesign(
            outcome='y', treatment='d', covariates=('x', 'x2')))


def test_perfect_treatment_separation_rejected_not_clipped():
    df = sample()
    df['d'] = (df.x > 0).astype(int)
    with pytest.raises((InsufficientSupportError, quasi.ConvergenceError)):
        quasi.estimate(df, design=design())


def test_missing_values_or_nonbinary_treatment_fail_fast():
    df = sample()
    df.loc[4, 'x'] = np.nan
    with pytest.raises(InvalidDesignError):
        quasi.estimate(df, design=design())
    df = sample()
    df.loc[4, 'd'] = 2
    with pytest.raises(InvalidDesignError):
        quasi.estimate(df, design=design('psm'))


def test_smd_uses_identical_before_after_denominator():
    df = sample()
    d = df.d.to_numpy()
    x = df[['x']].to_numpy()
    w = np.where(d == 0, 2 + np.maximum(df.x.to_numpy(), 0), 1.0)
    before, after = _balance(x, d, ['x'], w)
    s = np.sqrt((x[d == 1, 0].var(ddof=1) + x[d == 0, 0].var(ddof=1))/2)
    assert before['x'] == pytest.approx(abs(x[d == 1, 0].mean() - x[d == 0, 0].mean())/s)
    assert after['x'] == pytest.approx(abs(x[d == 1, 0].mean() - np.average(x[d == 0, 0], weights=w[d == 0]))/s)


def test_stacked_influence_se_matches_finite_difference_equations():
    df = sample(1200)
    y = df.y.to_numpy()
    d = df.d.to_numpy().astype(float)
    z = np.column_stack((np.ones(len(df)), df.x.to_numpy()))
    e = _propensity(z, d)
    # Recover the full logistic parameters using independent statsmodels,
    # and derive a numeric Jacobian of the joint moment equations.
    import statsmodels.api as sm
    glm = sm.GLM(d, z, family=sm.families.Binomial()).fit()
    beta, *_ = np.linalg.lstsq(z[d == 0], y[d == 0], rcond=None)
    odds = e/(1-e)
    att = np.sum((d-(1-d)*odds)*(y-z@beta))/d.sum()
    theta = np.r_[att, glm.params, beta]
    n, k = z.shape
    def eq(t):
        tau, gamma, b = t[0], t[1:k+1], t[k+1:]
        pr = expit(z @ gamma)
        rr = y - z @ b
        oo = (1-d)*pr/(1-pr)
        return np.column_stack([d*(rr-tau)-oo*rr, (d-pr)[:,None]*z, ((1-d)*rr)[:,None]*z])
    h = 1e-5
    jac_numeric = np.column_stack([(eq(theta + h*np.eye(len(theta))[j]).mean(axis=0) -
                                    eq(theta - h*np.eye(len(theta))[j]).mean(axis=0))/(2*h)
                                   for j in range(len(theta))])
    moments = eq(theta)
    influence = -np.linalg.solve(jac_numeric, moments.T)[0]
    expected = np.sqrt(np.mean((influence - influence.mean())**2)/n)
    r = quasi.estimate(df, design=design())
    assert r.estimate == pytest.approx(att, abs=2e-6)
    assert r.se == pytest.approx(expected, rel=1e-4)
    assert np.all(np.isfinite(jac_numeric))


def test_balance_diagnostics_are_not_identification_certificates():
    r = quasi.estimate(sample(), design=design())
    assert r.identification_status == 'not_established'
    assert r.model_meta['effective_control_weight_n'] > 0
    assert r.model_meta['propensity_clipped'] is False
    assert r.smd_before and r.smd_after


def test_nearest_neighbor_ties_are_deterministic_and_controls_reused():
    df = sample(700)
    r1 = quasi.PropensityScoreMatching(df, outcome='y', treatment='d', covariates=['x']).fit()
    r2 = quasi.PropensityScoreMatching(df, outcome='y', treatment='d', covariates=['x']).fit()
    assert_allclose(r1.matched_control_indices, r2.matched_control_indices)
    assert len(set(r1.matched_control_indices)) < len(r1.matched_control_indices)


def test_ipw_hajek_is_not_aipw_and_matches_weighted_means():
    df = sample(750)
    r = quasi.estimate(df, design=design('ipw'))
    d = df.d.to_numpy().astype(float)
    e = _propensity(np.column_stack([np.ones(len(df)), df.x.to_numpy()]), d)
    w = (1-d)*e/(1-e)
    expected = df.y.to_numpy()[d == 1].mean() - np.average(df.y.to_numpy(), weights=w)
    assert r.estimate == pytest.approx(expected)
    assert r.model_name == 'IPWATT'
    assert r.inference_details['method'] == 'stacked_logit_IPW_Hajek_HC0'
    assert r.se > 0
    assert quasi.ResultRecord.from_json(r.to_json()).model_name == 'IPWATT'
    assert quasi.estimate(df, outcome='y', method='ipw', treatment='d', covariates=('x',)).estimate == pytest.approx(expected)


def test_ipw_stacked_se_independent_finite_differences():
    from scipy.special import expit
    import statsmodels.api as sm
    df = sample(850)
    y = df.y.to_numpy()
    d = df.d.to_numpy().astype(float)
    z = np.column_stack([np.ones(len(df)), df.x.to_numpy()])
    gamma = sm.GLM(d, z, family=sm.families.Binomial()).fit().params
    e = expit(z@gamma)
    w = (1-d)*e/(1-e)
    t = np.r_[np.sum(d*y)/d.sum(), np.sum(w*y)/w.sum(), gamma]
    n,k = z.shape
    def eq(theta):
        mu1,mu0 = theta[:2]
        pr=expit(z@theta[2:]); odds=(1-d)*pr/(1-pr)
        return np.column_stack([d*(y-mu1), odds*(y-mu0), (d-pr)[:,None]*z])
    h=1e-5
    j = np.column_stack([(eq(t+h*np.eye(k+2)[i]).mean(axis=0)-eq(t-h*np.eye(k+2)[i]).mean(axis=0))/(2*h)
                         for i in range(k+2)])
    influence = -np.linalg.solve(j,eq(t).T)
    target = np.sqrt(np.mean(((influence[0]-influence[1])-
                               (influence[0]-influence[1]).mean())**2)/n)
    assert quasi.IPWATT(df, outcome='y', treatment='d', covariates=['x']).fit().se == pytest.approx(target, rel=1e-5)
