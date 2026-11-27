"""ITS phase 9C profile likelihood and estimation contracts; research certification OPEN."""
import numpy as np
import pandas as pd
import pytest
from scipy import linalg
import statsmodels.api as sm

import quasi
from quasi.errors import InvalidDesignError, InsufficientSupportError, NotIdentifiedError
from quasi.inference.serial import profile_ml_ar1


def make_data(seed=21, n=96, rho=0.62, intervention=48):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    z = np.column_stack((np.ones(n), t/100, (t>=intervention).astype(float),
                         np.maximum(t-intervention,0)/100))
    u=rng.normal(0, 0.45, n)
    e=np.empty(n)
    e[0]=u[0]/np.sqrt(1-rho**2)
    for i in range(1,n):
        e[i]=rho*e[i-1]+u[i]
    y=z@np.array([2,0.4,1.3,.2])+e
    return pd.DataFrame({'time':t,'y':y}), z, y


def test_exact_stationary_profile_loglik_matches_statsmodels_statespace():
    _,x,y=make_data()
    result=profile_ml_ar1(x,y)
    model=sm.tsa.SARIMAX(y,exog=x,order=(1,0,0),trend='n',
                         enforce_stationarity=True)
    external=model.fit(disp=False,maxiter=300)
    ours_params=np.r_[result.coefficients,result.rho,result.innovation_variance]
    # Independent state-space Kalman likelihood with a stationary initialization.
    ours_loglik=float(model.loglike(ours_params))
    assert ours_loglik==pytest.approx(float(external.llf),abs=1e-4)
    assert result.rho==pytest.approx(float(external.params[-2]),abs=2e-3)
    np.testing.assert_allclose(result.coefficients, external.params[:-2],rtol=2e-3,atol=3e-3)
    assert result.innovation_variance==pytest.approx(float(external.params[-1]),rel=2e-3)


def test_mle_covariance_independently_matches_gls_at_fitted_rho():
    _,x,y=make_data(rho=-.4)
    fit=profile_ml_ar1(x,y)
    sigma=linalg.toeplitz(fit.rho**np.arange(len(y)))
    independent=sm.GLS(y,x,sigma=sigma).fit()
    np.testing.assert_allclose(fit.coefficients,independent.params,rtol=0,atol=1e-9)
    np.testing.assert_allclose(fit.covariance,independent.cov_params(),rtol=1e-8,atol=1e-9)


def test_public_api_direct_design_legacy_and_distinct_metadata():
    df,_,_=make_data()
    base=dict(outcome='y',time='time',intervention_date=48,error_model='ar1_mle')
    direct=quasi.InterruptedTimeSeries(df,**base).fit()
    spec=quasi.estimate(df,design=quasi.ITSDesign(**base))
    legacy=quasi.estimate(df,method='its',**base)
    assert direct.estimate==pytest.approx(spec.estimate)
    assert legacy.estimate==pytest.approx(spec.estimate)
    assert direct.inference_details['method']=='AR1_stationary_Gaussian_profile_ML'
    assert direct.inference_details['reference_certified'] is False
    assert direct.inference_details['conditional_on_estimated_rho'] is True
    assert direct.inference_details['rho_estimation_uncertainty_accounted_for'] is False
    assert direct.identification_status=='not_established'
    assert np.isfinite(direct.effect_trajectory().select_dtypes('number').to_numpy()).all()
    prediction=direct.counterfactual_prediction()
    assert (prediction.prediction_lower<prediction.prediction_upper).all()
    assert (prediction.cumulative_prediction_se>0).all()


def test_error_model_mle_cannot_be_combined_with_hac_lag():
    df,_,_=make_data()
    with pytest.raises(InvalidDesignError):
        quasi.ITSDesign(outcome='y',time='time',intervention_date=48,error_model='ar1_mle',hac_lag=3)
    with pytest.raises(InvalidDesignError):
        quasi.InterruptedTimeSeries(df,'y','time',48,error_model='ar1_mle',hac_lag=3)


@pytest.mark.parametrize('tol',[-.1,0,1.0,float('nan'),True])
def test_profile_ml_invalid_tolerances(tol):
    _,x,y=make_data()
    with pytest.raises(InvalidDesignError):profile_ml_ar1(x,y,tolerance=tol)


def test_profile_ml_requires_full_rank_and_enough_observations():
    _,x,y=make_data()
    with pytest.raises(NotIdentifiedError): profile_ml_ar1(np.column_stack([x,x[:,1]]),y)
    with pytest.raises(InsufficientSupportError): profile_ml_ar1(x[:8],y[:8])


def test_negative_rho_and_seasonal_model_work_with_no_false_certification():
    df,_,_=make_data(seed=17,rho=-.54,n=120,intervention=60)
    df['y']=df['y']+np.cos(np.arange(120)*2*np.pi/12)
    r=quasi.InterruptedTimeSeries(df,'y','time',60,error_model='ar1_mle',seasonal_period=12).fit()
    assert -0.95 < r.inference_details['estimated_rho'] < 0
    assert r.inference_details['reference_certified'] is False
    assert r.se>0


def test_seeded_bootstrap_profile_mle_refits_method():
    df,_,_=make_data(seed=29)
    r=quasi.InterruptedTimeSeries(df,'y','time',48,error_model='ar1_mle').fit()
    orig=(r.ci_lower,r.ci_upper)
    a=r.ar1_bootstrap_level_interval(reps=51,seed=19)
    b=r.ar1_bootstrap_level_interval(reps=51,seed=19)
    assert a==b
    assert a['successful_refits']+a['failed_refits']==51
    assert a.get('error_model')=='ar1_mle'
    assert (r.ci_lower,r.ci_upper)==orig


def test_profile_ml_rejects_prais_winsten_only_iteration_setting():
    df,_,_=make_data()
    with pytest.raises(InvalidDesignError,match="ar1_max_iter"):
        quasi.ITSDesign(outcome="y",time="time",intervention_date=48,
                        error_model="ar1_mle",ar1_max_iter=40)
    with pytest.raises(InvalidDesignError,match="ar1_max_iter"):
        quasi.InterruptedTimeSeries(df,"y","time",48,
                                    error_model="ar1_mle",ar1_max_iter=40)


def test_legacy_prais_winsten_method_does_not_change():
    df,_,_=make_data()
    r=quasi.InterruptedTimeSeries(df,'y','time',48,error_model='ar1').fit()
    assert r.inference_details['method']=='Prais_Winsten_FGLS_AR1'
