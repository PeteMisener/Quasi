"""Algebraic RDD checks. These are not independent rdrobust certification."""
import numpy as np
import pandas as pd
import pytest
import quasi

from quasi.rdd.local_inference import boundary_weights, rd_inference


def rddata(seed=7, n=700, curve=0.7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = 3 + 1.4 * (x >= 0) + .55*x + curve*x*x * (x >= 0) + rng.normal(0, .6, n)
    return pd.DataFrame({'x': x, 'y': y, 'c': np.arange(n) % 22})


def test_sharp_sidewise_wls_and_manual_hc0():
    import statsmodels.api as sm
    df = rddata()
    x, y = df.x.to_numpy(), df.y.to_numpy()
    res = quasi.SharpRDD(df, outcome='y', running_var='x', cutoff=0.,
                         bandwidth=1.2, kernel='triangular').fit()
    fits = []
    for right in (False, True):
        sel = (x >= 0) if right else (x < 0)
        sel &= np.abs(x) < 1.2
        X = np.column_stack([np.ones(sel.sum()), x[sel]/1.2])
        w = 1 - np.abs(x[sel]/1.2)
        f = sm.WLS(y[sel], X, weights=w).fit(cov_type='HC0')
        fits.append(f)
    assert res.estimate == pytest.approx(fits[1].params[0] - fits[0].params[0], abs=1e-10)
    assert res.se == pytest.approx(np.sqrt(fits[1].cov_params()[0, 0] + fits[0].cov_params()[0, 0]), rel=1e-9)
    assert res.inference_details['method'] == 'conventional_hc0'
    assert res.inference_details['reference_certified'] is False


@pytest.mark.parametrize('order,kernel', [(1, 'uniform'), (1, 'epanechnikov'), (2, 'triangular')])
def test_sharp_alternative_order_and_kernel(order, kernel):
    df = rddata()
    result = quasi.estimate(df, design=quasi.RDDDesign(outcome='y', running_var='x',
                              cutoff=0, bandwidth=1.5, kernel=kernel, order=order))
    assert np.isfinite(result.se)
    assert 0 < result.p_value <= 1


def test_bias_corrected_recovers_quadratic_jump_noisily():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.5, 1.5, 1800)
    y = 1.0 + 2.5*(x >= 0) + 1.2*x + 4.5*(x >= 0)*x*x + rng.normal(0, .075, len(x))
    data = pd.DataFrame({'x': x, 'y': y})
    kwargs = dict(outcome='y', running_var='x', cutoff=0, bandwidth=1.35)
    conventional = quasi.SharpRDD(data, **kwargs).fit()
    corrected = quasi.SharpRDD(data, **kwargs, inference='bias_corrected', bias_bandwidth=1.5).fit()
    assert abs(corrected.estimate - 2.5) < abs(conventional.estimate - 2.5) / 3
    assert corrected.inference_details['conventional_estimate'] == pytest.approx(conventional.estimate)
    assert corrected.inference_details['bias_corrected_estimate'] == pytest.approx(corrected.estimate)
    assert corrected.inference_details['pilot_order'] == 2
    assert corrected.inference_details['reference_certified'] is False


def test_higher_order_correction_exactly_annihilates_quadratic_on_each_side():
    x = np.linspace(-1.5, 1.5, 601)
    r = boundary_weights(x, 1.1, 1.4, 1, 'triangular', 'right')
    l = boundary_weights(x, 1.1, 1.4, 1, 'triangular', 'left')
    L = r.corrected - l.corrected
    assert L @ np.ones_like(x) == pytest.approx(0., abs=1e-9)
    assert L @ x == pytest.approx(0., abs=1e-9)
    assert L @ (x*x) == pytest.approx(0., abs=1e-9)
    assert L @ (x >= 0).astype(float) == pytest.approx(1., abs=1e-9)


def test_fuzzy_joint_covariance_manual_delta():
    rng = np.random.default_rng(123)
    n = 2000
    x = rng.uniform(-1, 1, n)
    cutoff = x >= 0
    treatment = (rng.uniform(size=n) < np.where(cutoff, 0.85, 0.2)).astype(float)
    y = .7 + 1.7*treatment + .5*x + rng.normal(0, .7, n)
    r = rd_inference(x, y, .8, 'triangular', 1, treatment=treatment)
    left = boundary_weights(x, .8, None, 1, 'triangular', 'left')
    right = boundary_weights(x, .8, None, 1, 'triangular', 'right')
    L = right.conventional - left.conventional
    tau = float((L@y) / (L@treatment))
    assert r['estimate'] == pytest.approx(tau)
    assert abs(r['estimate']-1.7) < .2
    # Independent WLS residual fits of y-tau*T on each side, then HC0 delta variance.
    import statsmodels.api as sm
    influence = np.zeros(n)
    for mask in (x < 0, x >= 0):
        sub = mask & (np.abs(x) < .8)
        Z = np.column_stack([np.ones(sub.sum()), x[sub]/.8])
        w = 1 - np.abs(x[sub])/.8
        fit = sm.WLS((y-tau*treatment)[sub], Z, weights=w).fit()
        influence[sub] = L[sub] * fit.resid / float(L@treatment)
    assert r['se'] == pytest.approx(np.sqrt(influence@influence), rel=1e-10)


def test_fuzzy_and_sharp_designs_distinct():
    df = rddata()
    rng = np.random.default_rng(3)
    df['d'] = (rng.uniform(size=len(df)) < np.where(df.x >= 0, .8, .2)).astype(int)
    fuzzy = quasi.estimate(df, design=quasi.RDDDesign(outcome='y', running_var='x',
                cutoff=0, bandwidth=1.1, treatment='d'))
    assert fuzzy.is_fuzzy
    assert .3 < fuzzy.inference_details['first_stage_jump'] < .9


@pytest.mark.parametrize('spec', [
    {'inference':'unknown'}, {'inference':'conventional','bias_bandwidth':1.2},
    {'inference':'bias_corrected','bias_bandwidth':-1.},
    {'inference':'bias_corrected','bias_bandwidth':np.nan},
    {'inference':'bias_corrected','cluster':'c'},
])
def test_design_guardrails(spec):
    with pytest.raises((ValueError, NotImplementedError)):
        quasi.RDDDesign(outcome='y', running_var='x', cutoff=0, **spec)


def test_rank_deficiency_fail_closed():
    rng = np.random.default_rng(1)
    x = rng.choice([-.6, -.3, .3, .6], size=1200)
    y = 3 + 2*(x>=0)+rng.normal(size=len(x))
    df = pd.DataFrame({'x':x, 'y':y})
    with pytest.raises(ValueError, match='distinct running values|rank deficient'):
        quasi.SharpRDD(df,'y','x',0, bandwidth=1, order=2).fit()


def test_fuzzy_weak_first_stage_fail_closed():
    x = np.linspace(-1.,1.,801)
    # Smooth deterministic take-up with no jump: linear first stage exactly zero.
    y = 3 + (x>=0) + x + np.sin(x*17)*.01
    d = .2 + .08*x
    df = pd.DataFrame({'x':x,'y':y,'d':d})
    with pytest.raises(ValueError, match='first-stage'):
        quasi.FuzzyRDD(df, 'y', 'x', 0, treatment='d', bandwidth=.9).fit()


def test_clustered_conventional_requires_support():
    df = rddata(n=800)
    df['c'] = np.arange(len(df))%5
    with pytest.raises(ValueError, match='at least 10'):
        quasi.SharpRDD(df,'y','x',0,bandwidth=1.,cluster='c').fit()


def test_explicit_rdd_bias_corrected_router_and_metadata():
    df = rddata()
    res = quasi.estimate(df, design=quasi.RDDDesign(outcome='y',running_var='x',cutoff=0,
                         bandwidth=1.0,inference='bias_corrected',bias_bandwidth=1.5))
    assert res.to_metadata_dict()['inference_method'] == 'experimental_bias_corrected_hc0'
    assert 'not_established' == res.identification_status
    assert res.to_record().inference is not None


def test_alpha_90_interval_and_bandwidth_provenance():
    df=rddata()
    obj=quasi.SharpRDD(df,'y','x',0,alpha=.1).fit()
    assert obj.confidence_level == pytest.approx(.90)
    assert obj.inference_details['bandwidth_selection'] == 'provisional_heuristic_not_CCT'
    assert 'ci_95' not in obj.to_metadata_dict()['primary_metric']


def test_cutoff_ties_warn_without_claiming_identification():
    df=rddata(n=800)
    df.loc[:25,'x']=0.
    obj=quasi.SharpRDD(df,'y','x',0,bandwidth=1.).fit()
    assert 'Cutoff Mass Points' in obj.diagnostics
    assert obj.diagnostics['Cutoff Mass Points'].status == quasi.DiagnosticStatus.WARN
    assert obj.identification_status == 'not_established'


def test_first_stage_diagnostic_metadata_present():
    df=rddata(n=800)
    rng=np.random.default_rng(9)
    df['d']=(rng.uniform(size=len(df))<np.where(df.x>=0,.88,.1)).astype(int)
    res=quasi.FuzzyRDD(df,'y','x',0,treatment='d',bandwidth=1.).fit()
    assert res.inference_details['first_stage_se']>0
    assert 'Fuzzy First Stage' in res.diagnostics
    assert res.identification_status == 'not_established'


def test_bias_correction_requires_pilot_support():
    x=np.concatenate([np.repeat(-.5,40),np.repeat(-.25,40),np.repeat(.25,40),np.repeat(.5,40)])
    data=pd.DataFrame({'x':x,'y':1.+2*(x>=0)+np.random.default_rng(4).normal(0,.1,len(x))})
    with pytest.raises(ValueError,match='distinct running values|rank deficient'):
        quasi.SharpRDD(data,'y','x',0,bandwidth=1.,inference='bias_corrected',bias_bandwidth=1.1).fit()


def test_cluster_count_uses_effective_bandwidth_not_full_sample():
    df=rddata(n=3000)
    df['c'] = np.arange(len(df)) % 15
    df.loc[df.x.abs() < .4,'c'] = 0  # many clusters globally but one in the local window
    with pytest.raises(ValueError,match='per cutoff side'):
        quasi.SharpRDD(df,'y','x',0,bandwidth=.4,cluster='c').fit()
