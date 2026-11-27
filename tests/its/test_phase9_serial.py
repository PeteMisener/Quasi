"""Phase 9 numerical and inference contracts; NOT causal identification certification."""
import numpy as np
import pandas as pd
import pytest
from scipy import linalg, stats
import statsmodels.api as sm

import quasi
from quasi.errors import InvalidDesignError, InsufficientSupportError
from quasi.inference.serial import prais_winsten_ar1


def _data(seed=3, n=120, rho=0.65, level=2.0, slope=0.04):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = np.column_stack((np.ones(n), t, (t >= n//2).astype(float), np.maximum(0, t-n//2)))
    e = np.zeros(n)
    u = rng.standard_normal(n)
    e[0] = u[0] / np.sqrt(1-rho*rho)
    for i in range(1,n): e[i] = rho*e[i-1] + u[i]
    beta = [2, 0.025, level, slope]
    return pd.DataFrame({'t':t.astype(int), 'y':x@beta+e}), x


def test_prais_winsten_fixed_rho_gls_equivalence():
    df, x = _data()
    fit = prais_winsten_ar1(x, df.y.to_numpy())
    rho = fit.rho
    # An independent generalized least-squares calculation using the exact
    # stationary AR(1) correlation matrix (Prais-Winsten retains t=0).
    cov = linalg.toeplitz(rho ** np.arange(len(df)))
    independent = sm.GLS(df.y.to_numpy(), x, sigma=cov).fit()
    np.testing.assert_allclose(fit.coefficients, independent.params, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit.covariance, independent.cov_params(), rtol=1e-9, atol=1e-10)
    assert abs(rho) < .98
    assert fit.iterations > 0


def test_ar1_public_estimator_and_explicit_design_match():
    df, _ = _data()
    direct = quasi.InterruptedTimeSeries(df, 'y', 't', 60, error_model='ar1').fit()
    explicit = quasi.estimate(df, design=quasi.ITSDesign(outcome='y', time='t',
                                  intervention_date=60, error_model='ar1'))
    assert direct.estimate == pytest.approx(explicit.estimate)
    assert direct.se == pytest.approx(explicit.se)
    assert direct.inference_details['method'] == 'Prais_Winsten_FGLS_AR1'
    assert direct.inference_details['reference_certified'] is False
    assert direct.inference_details['rho_estimation_uncertainty_accounted_for'] is False
    assert direct.inference_details['degrees_freedom'] == len(df)-4
    assert 0 < direct.inference_details['estimated_rho'] < 0.98
    assert 'Prais_Winsten_FGLS_AR1' in direct.methods_text()
    assert np.isfinite(direct.to_frame().select_dtypes('number').to_numpy()).all()


def test_ar1_t_intervals_and_slope_p_value():
    df, _ = _data()
    r = quasi.InterruptedTimeSeries(df, 'y', 't', 60, error_model='ar1', alpha=.1).fit()
    dfree = r.inference_details['degrees_freedom']
    crit = stats.t.ppf(.95, dfree)
    assert r.ci_lower == pytest.approx(r.level_change - crit*r.level_se)
    assert r.slope_p == pytest.approx(2*stats.t.sf(abs(r.slope_change/r.slope_se),dfree))
    assert r.confidence_level == .9


def test_hac_default_numerical_backward_compatibility():
    df, _ = _data()
    a = quasi.InterruptedTimeSeries(df,'y','t',60,hac_lag=3).fit()
    b = quasi.InterruptedTimeSeries(df,'y','t',60,hac_lag=3,error_model='hac').fit()
    assert a.estimate == b.estimate and a.se == b.se
    assert a.inference_details['max_lag'] == 3
    assert 'HAC' in a.inference_details['method']


def test_covariance_respects_slope_and_cumulative_correlations():
    df, _ = _data()
    for mode in ('hac','ar1'):
        r = quasi.InterruptedTimeSeries(df,'y','t',60,error_model=mode,hac_lag=4 if mode=='hac' else None).fit()
        out = r.effect_trajectory()
        cov = r._coefficient_covariance
        c = np.array([0.,0.,3.,3.])  # Cumulative first 3 post contrasts: 1+1+1, 0+1+2
        expected_se = np.sqrt(c @ cov[:4,:4] @ c)
        assert out.cumulative_se.iloc[2] == pytest.approx(expected_se)
        assert out.effect.iloc[0] == pytest.approx(r.level_change)
        assert out.se.iloc[0] == pytest.approx(r.level_se)
        assert out.effect.iloc[3] == pytest.approx(r.level_change + 3*r.slope_change)
        assert out.cumulative_effect.iloc[-1] == pytest.approx(out.effect.sum())
        assert r.sub_effects['cumulative_post_effect'] == pytest.approx(out.cumulative_effect.iloc[-1])
        assert r.sub_effects['cumulative_post_se'] == pytest.approx(out.cumulative_se.iloc[-1])
        assert len(out)==len(df)-60
        assert np.isfinite(out.select_dtypes('number').to_numpy()).all()


def test_ar1_rejects_unsupported_options():
    df,_ = _data()
    with pytest.raises(InvalidDesignError):
        quasi.ITSDesign(outcome='y',time='t',intervention_date=60,error_model='ar1',hac_lag=3)
    with pytest.raises(InvalidDesignError):
        quasi.InterruptedTimeSeries(df,'y','t',60,error_model='ar1',hac_lag=2)
    with pytest.raises(InvalidDesignError):
        quasi.InterruptedTimeSeries(df,'y','t',60,error_model='ar2')
    with pytest.raises(InvalidDesignError):
        quasi.ITSDesign(outcome='y',time='t',intervention_date=60,error_model='ar1',ar1_max_iter=1)
    with pytest.raises(InvalidDesignError):
        quasi.ITSDesign(outcome='y',time='t',intervention_date=60,error_model='ar1',ar1_tolerance=0)
    with pytest.raises(InvalidDesignError):
        quasi.InterruptedTimeSeries(df,'y','t',60,error_model='ar1',ar1_tolerance=float('nan'))


def test_iteration_nonconvergence_refuses_to_report_result():
    df, x = _data()
    with pytest.raises(InsufficientSupportError, match='converge'):
        prais_winsten_ar1(x, df.y.to_numpy(), max_iter=2, tolerance=1e-16)


def test_direct_constructor_rejects_missing_cadence():
    df,_ = _data()
    df.loc[75:,'t'] += 1
    with pytest.raises(InvalidDesignError,match='regular'):
        quasi.InterruptedTimeSeries(df,'y','t',60).fit()
    with pytest.raises(InvalidDesignError,match='regular'):
        quasi.InterruptedTimeSeries(df,'y','t',60,error_model='ar1').fit()


def test_legacy_router_supports_explicit_error_model():
    df,_ = _data()
    r=quasi.estimate(df, outcome='y', method='its', time='t', intervention_date=60,
                     error_model='ar1')
    assert r.error_model=='ar1'


def test_trajectory_respects_alpha_and_does_not_assume_independent_periods():
    df,_=_data()
    r1=quasi.InterruptedTimeSeries(df,'y','t',60,error_model='ar1',alpha=.05).fit()
    r2=quasi.InterruptedTimeSeries(df,'y','t',60,error_model='ar1',alpha=.10).fit()
    x1=r1.effect_trajectory();x2=r2.effect_trajectory()
    assert (x1.ci_upper-x1.ci_lower).mean() > (x2.ci_upper-x2.ci_lower).mean()
    assert (x1.cumulative_ci_upper-x1.cumulative_ci_lower).mean() > (x2.cumulative_ci_upper-x2.cumulative_ci_lower).mean()
    assert r1.estimate == pytest.approx(r2.estimate)


def test_real_nile_dataset_against_statsmodels_gls_conditional_on_rho():
    data = sm.datasets.nile.load_pandas().data
    r = quasi.InterruptedTimeSeries(data, 'volume', 'year', 1921., error_model='ar1').fit()
    t = np.arange(len(data), dtype=float)
    x = np.column_stack((np.ones(len(t)), t, (t >= 50).astype(float), np.maximum(t-50,0)))
    rho = r.inference_details['estimated_rho']
    ref = sm.GLS(data.volume.values, x, sigma=linalg.toeplitz(rho**t)).fit()
    assert r.estimate == pytest.approx(float(ref.params[2]), abs=1e-8)
    assert r.se == pytest.approx(float(ref.bse[2]), abs=1e-8)
    assert r.slope_change == pytest.approx(float(ref.params[3]), abs=1e-8)
    assert r.slope_se == pytest.approx(float(ref.bse[3]), abs=1e-8)
    # This is conditional-on-Quasi-rho numerical equivalence, NOT an
    # independent rho estimator validation or causal interpretation.
    assert r.identification_status == 'not_established'


def test_ar1_diagnostics_use_whitened_innovations():
    frame, _ = _data(rho=.83)
    r = quasi.InterruptedTimeSeries(frame, 'y', 't', 60, error_model='ar1').fit()
    assert 'Durbin-Watson AR1 innovations' in r.diagnostics
    innovation_dw = r.diagnostics['Durbin-Watson AR1 innovations'].statistic_value
    raw = r.y_actual-r.y_fitted
    raw_dw = float(np.sum(np.diff(raw)**2)/np.sum(raw**2))
    assert innovation_dw > raw_dw  # autoregressive filtering removed dependence
    assert abs(2 - innovation_dw) < 1
    assert r.diagnostics['Durbin-Watson AR1 innovations'].p_value is None


def test_ar1_with_covariate_and_date_index():
    frame,_=_data()
    rng=np.random.default_rng(12)
    frame['cov'] = rng.normal(size=len(frame))
    frame['y'] += 1.75*frame['cov']
    frame['date']=pd.date_range('2010-01-01',periods=len(frame),freq='MS')
    intervention=frame.date.iloc[60]
    r=quasi.estimate(frame,design=quasi.ITSDesign(outcome='y',time='date',
        intervention_date=intervention,covariates=('cov',),error_model='ar1'))
    assert np.isfinite([r.estimate,r.se,r.slope_change]).all()
    assert r._coefficient_covariance.shape==(5,5)
    assert len(r.effect_trajectory())==60


def test_negligible_error_variation_does_not_invent_ar1_uncertainty():
    frame,_=_data(rho=0)
    t=np.arange(len(frame))
    frame['y']=3+0.07*t + (t>=60).astype(float)*2 + 0.03*np.maximum(0,t-60)
    with pytest.raises(InsufficientSupportError):
        quasi.InterruptedTimeSeries(frame,'y','t',60,error_model='ar1').fit()


def test_design_settings_serialize_and_conflicting_hac_lag_rejected():
    df,_=_data()
    design=quasi.ITSDesign(outcome='y',time='t',intervention_date=60,error_model='ar1')
    r=quasi.estimate(df,design=design)
    import json
    info=json.loads(r.to_json())
    assert info['inference']['method'] == 'Prais_Winsten_FGLS_AR1'
    assert info['inference']['independently_validated'] is False
    assert 'AR1' in r.summary() or 'AR1' in r.methods_text()
