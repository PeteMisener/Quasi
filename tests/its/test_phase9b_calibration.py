"""Phase 9B research safety regressions; NOT statistical certification."""
import numpy as np
import pandas as pd
import pytest
from scipy import linalg
import statsmodels.api as sm

import quasi
from quasi.errors import InvalidDesignError, InsufficientSupportError
from quasi.inference.serial import prais_winsten_ar1


def sample(seed=23, n=120, rho=.55, seasonal=True):
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    season = 2.0*np.cos(2*np.pi*t/12)+1.1*np.sin(2*np.pi*t/12) if seasonal else 0.
    innovation=rng.normal(0,.4,n)
    noise=np.empty(n)
    noise[0]=innovation[0]/np.sqrt(1-rho*rho)
    for k in range(1,n): noise[k]=rho*noise[k-1]+innovation[k]
    untreated=3+.025*t+season+noise
    y=untreated+1.2*(t>=60)+.015*np.maximum(t-60,0)
    return pd.DataFrame({'time':t, 'y':y}),untreated


def test_seasonal_fourier_matches_statsmodels_gls_conditional_on_rho():
    data, _=sample()
    r=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1',
       seasonal_period=12,seasonal_harmonics=2).fit()
    t=np.arange(len(data)); d=(t>=60).astype(float)
    x=np.column_stack([np.ones(len(t)), t, d, np.maximum(t-60,0),
       np.cos(2*np.pi*t/12),np.sin(2*np.pi*t/12),np.cos(4*np.pi*t/12),np.sin(4*np.pi*t/12)])
    rho=r.inference_details['estimated_rho']
    gls=sm.GLS(data.y, x, sigma=linalg.toeplitz(rho**np.arange(len(data)))).fit()
    assert r.estimate==pytest.approx(gls.params.iloc[2],abs=1e-8)
    assert r.se==pytest.approx(gls.bse.iloc[2],abs=1e-8)
    assert r.seasonal_period==12
    assert r.inference_details['seasonal_harmonics']==2
    assert r.identification_status=='not_established'


def test_nyquist_harmonic_omits_zero_sine_column():
    data,_=sample(n=120)
    r=quasi.InterruptedTimeSeries(data,'y','time',60,
        seasonal_period=2,seasonal_harmonics=1,error_model='ar1').fit()
    assert r._coefficient_covariance.shape == (5,5)


def test_seasonal_validation_direct_design_and_router():
    data,_=sample()
    kw=dict(outcome='y',time='time',intervention_date=60,
       seasonal_period=12,seasonal_harmonics=1)
    direct=quasi.InterruptedTimeSeries(data,**kw).fit()
    explicit=quasi.estimate(data,design=quasi.ITSDesign(**kw))
    legacy=quasi.estimate(data, method='its',**kw)
    assert direct.estimate==pytest.approx(explicit.estimate)
    assert direct.estimate==pytest.approx(legacy.estimate)
    assert 'Ljung-Box residual screen' in direct.diagnostics
    assert 'Ljung-Box residual screen' in explicit.diagnostics


@pytest.mark.parametrize('period,harmonics',[(0,1),(12,0),(12,7),(12,1.1),(12,True),('12',1), (None,2)])
def test_invalid_fourier_spec_rejected(period,harmonics):
    data,_=sample()
    kw=dict(outcome='y',time='time',intervention_date=60,
       seasonal_period=period,seasonal_harmonics=harmonics)
    with pytest.raises(InvalidDesignError): quasi.ITSDesign(**kw)
    with pytest.raises(InvalidDesignError): quasi.InterruptedTimeSeries(data,**kw)


def test_insufficient_seasonal_cycles_refused():
    data,_=sample(n=120)
    with pytest.raises(InsufficientSupportError,match='cycles'):
        quasi.InterruptedTimeSeries(data,'y','time',60,seasonal_period=36).fit()


def test_counterfactual_prediction_one_step_and_cumulative_formula():
    data,_=sample()
    r=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1',seasonal_period=12).fit()
    pred=r.counterfactual_prediction()
    info=r._forecast_inputs
    rho=info['rho']; sigma2=info['innovation_variance']; beta=info['coefficients']
    xcf=info['counterfactual_design']; xpre=info['last_pre_design']; yp=info['last_pre_y']
    cov=r._coefficient_covariance
    expected_mean=float(xcf[0]@beta+rho*(yp-xpre@beta))
    first_row=xcf[0]-rho*xpre
    expected_var=first_row@cov@first_row+sigma2
    assert pred.counterfactual_mean.iloc[0]==pytest.approx(expected_mean)
    assert pred.prediction_se.iloc[0]**2==pytest.approx(expected_var)
    # Explicit future-innovation covariance via matrix of AR impulse weights.
    m=5
    steps=np.arange(1,m+1)
    effective= xcf[:m]-rho**steps[:,None]*xpre
    L=np.tril(rho**np.subtract.outer(np.arange(m),np.arange(m)))
    expected_cov=effective@cov@effective.T+sigma2*(L@L.T)
    assert pred.cumulative_prediction_se.iloc[m-1]**2==pytest.approx(expected_cov.sum())
    assert pred.cumulative_counterfactual_mean.iloc[m-1]==pytest.approx(pred.counterfactual_mean.iloc[:m].sum())
    assert np.all(pred.prediction_lower < pred.prediction_upper)
    assert pred.prediction_se.iloc[0] > 0
    # Effect mean contrast and counterfactual predictive distribution are not interchangeable.
    eff=r.effect_trajectory()
    assert pred.prediction_se.iloc[0] > eff.se.iloc[0]


def test_hac_prediction_not_silently_labeled_ar1():
    data,_=sample()
    r=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='hac').fit()
    with pytest.raises(InvalidDesignError,match='AR1'):
        r.counterfactual_prediction()


def test_seasonal_diagnostics_provide_descriptive_multilag_screen():
    # Omitted periodicity should create residual autocorrelation at longer lags.
    data,_=sample(rho=.3)
    omitted=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1').fit()
    modeled=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1',seasonal_period=12).fit()
    assert omitted.diagnostics['Ljung-Box residual screen'].p_value < .05
    assert modeled.diagnostics['Ljung-Box residual screen'].p_value > omitted.diagnostics['Ljung-Box residual screen'].p_value
    assert modeled.inference_details['reference_certified'] is False


def test_ar1_bootstrap_reestimates_rho_is_seeded_and_keeps_default_ci():
    data,_=sample()
    r=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1',seasonal_period=12).fit()
    original=(r.ci_lower,r.ci_upper)
    one=r.ar1_bootstrap_level_interval(reps=65,seed=90)
    two=r.ar1_bootstrap_level_interval(reps=65,seed=90)
    assert one==two
    assert one['successful_refits']+one['failed_refits']==65
    assert one['rho_reestimated_each_replication']
    assert one['reference_certified'] is False
    assert np.isfinite([one['ci_lower'],one['ci_upper']]).all()
    assert one['ci_lower']<one['ci_upper']
    assert (r.ci_lower,r.ci_upper)==original


def test_bootstrap_rejects_invalid_args_and_hac():
    data,_=sample()
    ar=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1').fit()
    hac=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='hac').fit()
    with pytest.raises(InvalidDesignError, match='AR1'):
        hac.ar1_bootstrap_level_interval()
    for kwargs in ({'reps':12},{'reps':True},{'seed':-1},{'seed':2.5}):
        with pytest.raises(InvalidDesignError):ar.ar1_bootstrap_level_interval(**kwargs)


def test_bootstrap_failure_gate_cannot_report_false_interval(monkeypatch):
    from quasi.its import segmented
    data,_=sample()
    fitted=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1').fit()
    def fail(*args,**kwargs):
        raise InsufficientSupportError('nonconvergence')
    monkeypatch.setattr(segmented,'prais_winsten_ar1',fail)
    with pytest.raises(InsufficientSupportError,match='Bootstrap failed'):
        fitted.ar1_bootstrap_level_interval(reps=50,seed=4)


def test_prediction_includes_innovation_variance_under_negative_ar1():
    data,_=sample(rho=-.45)
    fitted=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1').fit()
    pred=fitted.counterfactual_prediction()
    assert (pred.prediction_se>0).all()
    assert (pred.cumulative_prediction_se>0).all()
    assert np.isfinite(pred.select_dtypes('number').to_numpy()).all()
    assert pred.cumulative_prediction_se.iloc[5] > pred.cumulative_prediction_se.iloc[0]
