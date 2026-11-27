"""Phase 8 algebraic and adverse-case tests. None certify causal exclusion."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import quasi
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.iv.inference import anderson_rubin_test
from quasi.inference import cluster_covariance, hc_covariance


def frame(seed=304, *, n=600, q=2, clusters=30, instrument_strength=0.9, correlated=True):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(clusters), n//clusters)
    n = len(ids)
    z = rng.normal(size=(n, q))
    w = rng.normal(size=n)
    v = rng.normal(size=n)
    group = rng.normal(size=clusters)[ids] if correlated else np.zeros(n)
    x = instrument_strength * z[:, 0] + 0.3*w + v + 0.25*group
    y = 1.4*x + 0.6*w + 0.65*v + 0.7*group + rng.normal(scale=0.6+0.4*abs(w), size=n)
    df = pd.DataFrame({'y':y, 'x':x, 'w':w, 'grp':ids})
    for j in range(q):
        df[f'z{j}'] = z[:,j]
    return df


def matrices(df, q=2):
    n = len(df)
    x = np.column_stack([np.ones(n), df.x, df.w])
    z = np.column_stack([np.ones(n), *[df[f'z{j}'] for j in range(q)], df.w])
    return x,z


def test_coefficient_and_hc_covariance_agree_independent_iv2sls_and_manual_sandwich():
    from statsmodels.sandbox.regression.gmm import IV2SLS
    df = frame()
    x,z = matrices(df)
    ref = IV2SLS(df.y.to_numpy(), x, z).fit()
    out = quasi.TwoStageLS(df,'y','x',['z0','z1'],covariates=['w']).fit()
    assert out.estimate == pytest.approx(ref.params[1], abs=1e-10)
    projected = z @ np.linalg.lstsq(z,x,rcond=None)[0]
    u = df.y.to_numpy()-x @ ref.params
    bread = np.linalg.inv(projected.T @ projected)
    manual = bread @ (projected*u[:,None]).T @ (projected*u[:,None]) @ bread
    assert out.se**2 == pytest.approx(manual[1,1], rel=1e-9)
    assert out.inference_details['covariance']=='HC0'
    assert out.sargan_j_p is None  # homoskedastic Sargan cannot be used under HC0
    assert 'Overidentification not assessed' in out.diagnostics


def test_hc1_correction_and_design_routing():
    df = frame(q=1)
    hc0 = quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w']).fit()
    hc1 = quasi.estimate(df, design=quasi.IVDesign(outcome='y',endogenous='x',instruments=('z0',),covariates=('w',),covariance='HC1'))
    assert hc1.estimate == pytest.approx(hc0.estimate)
    assert hc1.se**2 / hc0.se**2 == pytest.approx(len(df)/(len(df)-3),rel=1e-10)
    assert hc1.to_record().inference.method=='IV_2SLS_HC1'
    assert hc1.to_record().inference.independently_validated is False


def test_cluster_cr1_covariance_matches_reference_sandwich_and_t_reference():
    df = frame()
    out=quasi.estimate(df,design=quasi.IVDesign(outcome='y',endogenous='x',instruments=('z0','z1'),covariates=('w',),cluster='grp'))
    x,z = matrices(df)
    projected = z @ np.linalg.lstsq(z,x,rcond=None)[0]
    u = df.y.to_numpy()-x @ np.linalg.lstsq(projected,df.y.to_numpy(),rcond=None)[0]
    bread = np.linalg.inv(projected.T @ projected)
    scores=np.array([(projected[df.grp==g]*u[df.grp==g,None]).sum(axis=0) for g in df.grp.unique()])
    g = df.grp.nunique()
    factor = (g/(g-1)) * (len(df)-1)/(len(df)-x.shape[1])
    variance=(bread @ (scores.T@scores)*factor @ bread)[1,1]
    assert out.se**2 == pytest.approx(variance, rel=2e-10)
    assert out.inference_details['clusters'] == g
    assert out.inference_details['reference_distribution'] == 't'
    assert out.p_value == pytest.approx(2*stats.t.sf(abs(out.estimate/out.se),g-1))
    assert out.ci_upper-out.estimate == pytest.approx(stats.t.ppf(.975,g-1)*out.se)
    assert out.first_stage_robust.distribution=='F'
    assert out.ar_test(1.4).df_denominator==g-1


def test_ar_matches_independent_statsmodels_robust_ols_test():
    import statsmodels.api as sm
    df = frame(q=2)
    _,z=matrices(df)
    test_value=1.4
    ols=sm.OLS(df.y-test_value*df.x,z).fit(cov_type='HC0')
    b=np.asarray(ols.params)[1:3]
    v=np.asarray(ols.cov_params())[1:3,1:3]
    w=float(b @ np.linalg.solve(v,b))
    got=quasi.TwoStageLS(df,'y','x',['z0','z1'],covariates=['w']).fit().ar_test(test_value)
    assert got.statistic == pytest.approx(w, rel=1e-9)
    assert got.p_value == pytest.approx(stats.chi2.sf(w,2))
    assert got.distribution=='chi2'


def test_ar_null_zero_is_serialized_and_not_a_wald_confidence_set():
    df=frame(q=1)
    r=quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w']).fit()
    assert r.ar_null_zero.p_value == pytest.approx(r.ar_test(0.).p_value)
    assert r.ar_test(1.4).p_value > r.ar_null_zero.p_value
    assert r.inference_details['weak_instrument_robust_ci'] is False
    data=r.to_record().to_dict()
    assert data['additional_metadata']['inference_settings']['weak_instrument_robust_ci'] is False
    assert data['experimental'] is True


def test_robust_first_stage_partial_r2_and_reference_wald():
    import statsmodels.api as sm
    df=frame(q=2)
    _,z=matrices(df)
    first=sm.OLS(df.x,z).fit(cov_type='HC0')
    robust=quasi.TwoStageLS(df,'y','x',['z0','z1'],covariates=['w']).fit()
    coef=np.asarray(first.params)[1:3]
    cov=np.asarray(first.cov_params())[1:3,1:3]
    w=float(coef@np.linalg.solve(cov,coef))
    assert robust.first_stage_robust.statistic == pytest.approx(w,rel=1e-9)
    ssrrest=np.sum(sm.OLS(df.x,z[:,[0,3]]).fit().resid**2)
    ssrfull=np.sum(first.resid**2)
    assert robust.first_stage_partial_r2 == pytest.approx((ssrrest-ssrfull)/ssrrest)
    assert robust.diagnostics['First-stage conditional relevance'].status.value=='WARN'


def test_overidentified_homoskedastic_sargan_not_mislabeled_as_hansen_j():
    r=quasi.TwoStageLS(frame(),'y','x',['z0','z1'],covariates=['w']).fit()
    assert r.sargan_j_p is None
    assert all('Hansen' not in s.name for s in r.diagnostics.values())


@pytest.mark.parametrize('kwargs',[
    {'covariance':'XYZ'},
    {'covariance':'CR1'},
    {'cluster':'grp','covariance':'HC0'},
    {'cluster':'grp','covariance':'HC1'},
])
def test_inference_options_rejected_early(kwargs):
    df=frame(q=1)
    with pytest.raises((InvalidDesignError, ValueError),match='covariance'):
        quasi.TwoStageLS(df,'y','x',['z0'],**kwargs)


def test_cluster_too_few_and_all_same_fail():
    df=frame(q=1)
    df['grp']=np.arange(len(df))//120
    with pytest.raises(InsufficientSupportError, match='at least 10'):
        quasi.TwoStageLS(df,'y','x',['z0'],cluster='grp').fit()


def test_weak_first_stage_not_certified_as_strong_and_ar_finite():
    r=quasi.TwoStageLS(frame(instrument_strength=0.015,q=1),'y','x',['z0'],covariates=['w']).fit()
    assert np.isfinite(r.first_stage_robust.statistic)
    assert 0 <= r.ar_test(1.4).p_value <=1
    assert not r.passed_validation


def test_ar_invalid_null_rejected():
    r=quasi.TwoStageLS(frame(q=1),'y','x',['z0'],covariates=['w']).fit()
    for x in (float('nan'), float('inf'), True, None):
        with pytest.raises((InvalidDesignError, ValueError, TypeError)):
            r.ar_test(x)


def test_missing_covariance_with_cluster_routes_cr1():
    df=frame(q=1)
    direct=quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w'],cluster='grp').fit()
    routed=quasi.estimate(df,'y',method='iv',endogenous='x',instruments=('z0',),covariates=('w',),cluster='grp')
    assert direct.se == pytest.approx(routed.se)
    assert direct.covariance == routed.covariance == 'CR1'


def test_missing_cluster_labels_fail_closed():
    df=frame(q=1);df.loc[1,'grp']=np.nan
    with pytest.raises((InvalidDesignError,ValueError)):
        quasi.estimate(df,design=quasi.IVDesign(outcome='y', endogenous='x',instruments=('z0',),cluster='grp'))


def test_iv_experiment_is_seeded_and_explicitly_uncertified():
    from validation.iv_monte_carlo import run
    a=run(reps=4,n=60,seed=14)
    b=run(reps=4,n=60,seed=14)
    assert a==b
    assert not a['reference_certified']
    assert len(a['scenarios'])==2


def test_linearmodels_reference_fixture_is_hash_bound_and_run_fails_closed(monkeypatch,tmp_path):
    import hashlib
    from pathlib import Path
    from validation import iv_linearmodels_parity
    fixture=Path(iv_linearmodels_parity.__file__).parent/'fixtures'/'iv_phase8_synthetic.csv'
    expected=(fixture.with_suffix('.sha256')).read_text().strip()
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==expected
    # Missing reference package is an explicit failure, never a fake PASS fixture.
    import builtins
    real_import=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.startswith('linearmodels'):
            raise ImportError('blocked independent reference for negative control')
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    with pytest.raises(RuntimeError, match='independent linearmodels'):
        iv_linearmodels_parity.run()
    assert not (tmp_path/'reference_pass.json').exists()


def test_clustered_ar_matches_independent_cluster_robust_auxiliary_ols():
    import statsmodels.api as sm
    df=frame(q=2)
    _,z=matrices(df)
    beta0=1.4
    aux=sm.OLS(df.y-beta0*df.x,z).fit(cov_type='cluster',cov_kwds={'groups':df.grp,
                             'use_correction':True},use_t=False)
    b=np.asarray(aux.params)[1:3]
    cov=np.asarray(aux.cov_params())[1:3,1:3]
    w=float(b@np.linalg.solve(cov,b))/2
    out=quasi.TwoStageLS(df,'y','x',['z0','z1'],covariates=['w'],cluster='grp').fit()
    assert out.ar_test(beta0).statistic==pytest.approx(w,rel=2e-9)
    assert out.ar_test(beta0).p_value==pytest.approx(stats.f.sf(w,2,df.grp.nunique()-1))


def test_clustered_first_stage_robust_matches_statsmodels_wald():
    import statsmodels.api as sm
    df=frame(q=2)
    _,z=matrices(df)
    first=sm.OLS(df.x,z).fit(cov_type='cluster',cov_kwds={'groups':df.grp,
                             'use_correction':True},use_t=False)
    b=np.asarray(first.params)[1:3]
    cov=np.asarray(first.cov_params())[1:3,1:3]
    w=float(b@np.linalg.solve(cov,b))/2
    got=quasi.TwoStageLS(df,'y','x',['z0','z1'],covariates=['w'],cluster='grp').fit()
    assert got.first_stage_robust.statistic==pytest.approx(w,rel=2e-9)


def test_explicit_cr0_without_small_sample_adjustment():
    df=frame(q=1)
    a=quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w'],cluster='grp',covariance='CR0').fit()
    b=quasi.TwoStageLS(df,'y','x',['z0'],covariates=['w'],cluster='grp',covariance='CR1').fit()
    g=df.grp.nunique(); n=len(df)
    assert b.se**2/a.se**2==pytest.approx(g/(g-1)*(n-1)/(n-3))
    assert a.inference_details['small_sample'] is False
    assert b.inference_details['small_sample'] is True
