"""Iteration 15: independent AR-set inversion and auxiliary GMM J checks.

These establish algebra/engineering correctness only, not statistical certification.
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

import quasi
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.iv.inference import _quadratic_nonpositive, anderson_rubin_confidence_set
from tests.iv.test_phase8_inference import frame, matrices


@pytest.mark.parametrize('kind,cluster', [('HC0',False),('HC1',False),('CR0',True),('CR1',True)])
def test_analytic_inversion_agrees_with_independent_wald_null_tests(kind,cluster):
    df=frame(q=1,seed=31)
    r=quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w'],
                       cluster='grp' if cluster else None,covariance=kind).fit()
    cs=r.ar_confidence_set()
    _,z=matrices(df,q=1)
    for beta in np.linspace(-5,7,41):
        ols=sm.OLS(df.y-beta*df.x,z)
        if cluster:
            fit=ols.fit(cov_type='cluster',cov_kwds={'groups':df.grp,'use_correction':kind=='CR1'},use_t=False)
            stat=float(fit.params.iloc[1]**2/fit.cov_params().iloc[1,1])
            accepted=stat <= stats.f.ppf(.95,1,df.grp.nunique()-1)
        else:
            fit=ols.fit(cov_type=kind)
            stat=float(fit.params.iloc[1]**2/fit.cov_params().iloc[1,1])
            accepted=stat <= stats.chi2.ppf(.95,1)
        assert cs.contains(float(beta)) == accepted
        assert cs.contains(float(beta)) == (r.ar_test(float(beta)).p_value >= .05)
    assert cs.covariance == kind
    assert cs.level == pytest.approx(.95)


def test_weak_instrument_sets_can_be_all_real_not_truncated_to_search_range():
    df=frame(q=1,seed=305,instrument_strength=.015)
    result=quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w']).fit()
    cs=result.ar_confidence_set()
    assert cs.is_unbounded
    assert cs.contains(-1e12) and cs.contains(1e12)
    assert result.inference_details['weak_instrument_robust_ci'] is False
    assert result.inference_details['ar_set_available_for_single_instrument'] is True


def test_quadratic_inversion_all_possible_shapes():
    assert _quadratic_nonpositive(1,0,-1)==((-1.,1.),)
    assert _quadratic_nonpositive(-1,0,1)==((-np.inf,-1.),(1.,np.inf))
    assert _quadratic_nonpositive(1,0,1)==()
    assert _quadratic_nonpositive(-1,0,-1)==((-np.inf,np.inf),)
    assert _quadratic_nonpositive(0,1,-2)==((-np.inf,2.),)
    assert _quadratic_nonpositive(0,-1,-2)==((-2.,np.inf),)
    assert _quadratic_nonpositive(1,-2,1)==((1.,1.),)


def test_ar_set_alpha_changes_and_is_not_wald_ci():
    df=frame(q=1)
    r=quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w']).fit()
    s95=r.ar_confidence_set()
    s90=r.ar_confidence_set(alpha=.1)
    assert s95.intervals[0][0] < s90.intervals[0][0]
    assert s95.intervals[0][1] > s90.intervals[0][1]
    assert s95.intervals[0] != (r.ci_lower,r.ci_upper)


@pytest.mark.parametrize('bad', [0,1,np.nan,np.inf,True])
def test_ar_set_invalid_alpha_rejected(bad):
    r=quasi.TwoStageLS(frame(q=1),'y','x',['z0'],covariates=['w']).fit()
    with pytest.raises(InvalidDesignError):
        r.ar_confidence_set(alpha=bad)


def test_overidentified_ar_inversion_not_grid_approximated():
    r=quasi.TwoStageLS(frame(q=2),'y','x',['z0','z1'],covariates=['w']).fit()
    with pytest.raises(InvalidDesignError, match='one excluded instrument'):
        r.ar_confidence_set()


def test_gmm_j_matches_independent_numeric_minimization_of_moment_objective():
    df=frame(q=2,seed=41)
    r=quasi.TwoStageLS(df,'y','x',['z0','z1'],covariates=['w']).fit()
    t=r.gmm_overidentification_test()
    x,z=matrices(df,q=2)
    n=len(df)
    y=df.y.to_numpy()
    from statsmodels.sandbox.regression.gmm import IV2SLS
    b0=IV2SLS(y,x,z).fit().params
    scores=z*(y-x@b0)[:,None]
    S=scores.T@scores/n
    W=np.linalg.inv(S)
    def objective(beta):
        m=z.T@(y-x@beta)/n
        return n*float(m@W@m)
    ref=minimize(objective,np.asarray(b0),method='BFGS',tol=1e-8)
    assert t.statistic==pytest.approx(ref.fun,rel=1e-6,abs=1e-6)
    assert t.degrees_of_freedom == 1
    assert t.p_value==pytest.approx(stats.chi2.sf(t.statistic,1))
    assert t.gmm_coefficients[1]==pytest.approx(ref.x[1],rel=1e-5)
    assert r.sargan_j_p is None  # No automatic replacement of 2SLS diagnostic


def test_gmm_j_rejects_deliberately_invalid_extra_instrument():
    df=frame(q=2,n=3000,clusters=30)
    df['z1']=df.z1+0.4*df.y  # invalid instrument, correlated with structural error
    r=quasi.TwoStageLS(df,'y','x',['z0','z1'],covariates=['w']).fit()
    t=r.gmm_overidentification_test()
    assert t.p_value < .01
    assert t.statistic > 0


def test_auxiliary_j_guardrails_and_no_automatic_certification():
    r=quasi.TwoStageLS(frame(q=1),'y','x',['z0'],covariates=['w']).fit()
    with pytest.raises(InvalidDesignError,match='more instruments'):
        r.gmm_overidentification_test()
    c=quasi.TwoStageLS(frame(q=2),'y','x',['z0','z1'],covariates=['w'],cluster='grp').fit()
    with pytest.raises(InvalidDesignError,match='Clustered'):
        c.gmm_overidentification_test()
    assert not c.passed_validation


def test_analytic_ar_inversion_score_singularity_is_not_silently_accepted():
    rng=np.random.default_rng(4)
    z=rng.normal(size=90); Z=np.column_stack((np.ones(90),z))
    x=rng.normal(size=90)
    y=2*x+1.5*z  # residual scores are exactly proportional
    with pytest.raises(InsufficientSupportError,match='vanishes'):
        anderson_rubin_confidence_set(y,x,Z,1)


def test_phase8b_monte_carlo_is_seeded_and_uncertified():
    from validation.iv_advanced_monte_carlo import run
    a=run(reps=5,n=60,seed=15)
    assert a == run(reps=5,n=60,seed=15)
    assert a['reference_certified'] is False
    assert set(a['scenarios']) == {'valid_weak','valid_strong','invalid_exclusion'}
    assert all(s['failed_replications']==0 for s in a['scenarios'].values())


def test_real_data_generating_design_can_yield_disconnected_ar_set():
    r=quasi.TwoStageLS(frame(seed=3,n=200,clusters=20,q=1,instrument_strength=0.),
                       'y','x',['z0'],covariates=['w']).fit()
    s=r.ar_confidence_set()
    assert s.is_disconnected and s.is_unbounded
    assert s.contains(-1e5) and s.contains(1e5)
    assert not s.contains((s.intervals[0][1]+s.intervals[1][0])/2)
