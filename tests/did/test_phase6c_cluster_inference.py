"""Phase 6C: independently checked cluster score algebra and adverse support.

These tests verify *implementation* of cluster influence aggregation. They do
NOT establish equivalence to R did / appropriate finite-cluster coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from quasi import PanelDesign, StaggeredDiD, estimate
from quasi.did.inference import InferencePlan


def clustered_panel(seed=71, n_clusters=24, per_cluster=7, effect=1.25):
    rng = np.random.default_rng(seed)
    rows = []
    # each cluster's treatment assignment is consistent within cluster,
    # but cluster-level post shocks induce positive dependence across units.
    for g in range(n_clusters):
        treated = g < n_clusters//2
        post_shock = rng.normal(0, .9)
        for j in range(per_cluster):
            u = g*per_cluster + j
            intercept = rng.normal()
            idiosyncratic = rng.normal(0, .2)
            for t in range(3):
                y = intercept + .3*t + (post_shock + idiosyncratic + effect*treated if t == 2 else 0)
                rows.append({'unit':u, 'year':t, 'cohort': 2 if treated else -1,
                             'outcome':y, 'region':f'region_{g:03}'})
    return pd.DataFrame(rows)


def model(data, **kwargs):
    return StaggeredDiD(data, outcome='outcome', unit='unit', time='year',
                        first_treated='cohort', b_reps=79, seed=27, **kwargs).fit()


def test_cluster_score_covariance_matches_direct_independent_cluster_sum():
    n, m = 120, 20
    labels = np.repeat(np.arange(m), n//m)
    rng = np.random.default_rng(31)
    psi = rng.normal(size=(n, 3))
    plan = InferencePlan.build(labels, n)
    manual = np.stack([psi[labels == cluster].sum(axis=0) for cluster in range(m)])
    expected = m/(m-1) * manual.T @ manual / n**2
    assert plan.covariance(psi) == pytest.approx(expected, abs=1e-14)
    assert plan.se(psi[:,0]) == pytest.approx(np.sqrt(expected[0,0]), abs=1e-14)
    assert plan.critical(.05) == pytest.approx(stats.t.ppf(.975, df=m-1))
    assert plan.tail_probability(2.3) == pytest.approx(2*stats.t.sf(2.3, m-1))


def test_unclustered_score_covariance_retains_previous_unit_formula():
    rng = np.random.default_rng(44)
    psi = rng.normal(size=(62, 2))
    plan = InferencePlan.build(None, 62)
    assert plan.covariance(psi) == pytest.approx(psi.T @ psi/62**2)
    assert plan.critical(.05) == pytest.approx(stats.norm.ppf(.975))


def test_coarse_clustered_2x2_did_matches_statsmodels_cluster_sandwich():
    # For one ATT(2,2) cell, the group-mean DiD is OLS of y(2)-y(1) on
    # intercept + treated. Its uncorrected sandwich covariance is a
    # separate algebraic oracle for Quasi's cluster IF aggregation.
    import statsmodels.api as sm
    df = clustered_panel()
    r = model(df, cluster='region')
    raw = df.pivot(index='unit', columns='year', values='outcome')
    treat = df.drop_duplicates('unit').set_index('unit').reindex(raw.index)['cohort'].eq(2).astype(int)
    c = df.drop_duplicates('unit').set_index('unit').reindex(raw.index)['region']
    y = raw[2] - raw[1]
    fit = sm.OLS(y, sm.add_constant(treat)).fit()
    cov = fit.get_robustcov_results(cov_type='cluster', groups=c, use_correction=False,
                                    df_correction=False, use_t=False)
    uncorrected_se = cov.bse[1]
    correction = np.sqrt(r.inference_details['cluster_variance_correction'])
    assert r.estimate == pytest.approx(fit.params['cohort'], abs=1e-12)
    assert r.se == pytest.approx(uncorrected_se*correction, rel=1e-9)
    assert r.inference_details['degrees_of_freedom'] == 23
    assert r.model_meta['cluster'] == 'region'
    assert r.inference_details['se_unit'] == 'cluster'
    assert r.group_time_frame()['n_treated_clusters'].min() == 12
    assert r.group_time_frame()['n_control_clusters'].min() == 12
    assert r.p_value == pytest.approx(2*stats.t.sf(abs(r.estimate/r.se), 23))
    assert (r.ci_upper - r.estimate)/r.se == pytest.approx(stats.t.ppf(.975, 23))
    assert r.to_record().experimental


def test_clustered_estimation_inflates_uncertainty_when_data_cluster_correlated():
    df=clustered_panel()
    r_unit=model(df)
    r_cluster=model(df, cluster='region')
    assert r_cluster.estimate == pytest.approx(r_unit.estimate)
    assert r_cluster.se > r_unit.se * 1.5
    spec=PanelDesign(outcome='outcome', unit='unit', time='year', first_treated='cohort',
                     cluster='region', b_reps=79, seed=27)
    assert estimate(df, design=spec).se == pytest.approx(r_cluster.se)
    assert estimate(df, 'outcome', method='did', unit='unit', time='year',
                    first_treated='cohort', cluster='region', b_reps=79, seed=27).se == pytest.approx(r_cluster.se)


def test_cluster_multiplier_deterministic_and_uniform_bands_not_narrower_than_pointwise():
    a=model(clustered_panel(), cluster='region')
    b=model(clustered_panel().sample(frac=1, random_state=23), cluster='region')
    assert a.se == pytest.approx(b.se)
    assert a.simultaneous_ci == pytest.approx(b.simultaneous_ci)
    tcrit=stats.t.ppf(.975, 23)
    for e, center in a.event_study_effects.items():
        if a.event_study_se[e] > 0:
            assert a.simultaneous_ci[e][1] - center >= tcrit*a.event_study_se[e] - 1e-12


def test_noninvariant_cluster_ids_missing_clusters_and_low_cluster_count_fail():
    base=clustered_panel()
    wrong=base.copy()
    wrong.loc[(wrong.unit==0) & (wrong.year==1), 'region']='elsewhere'
    with pytest.raises(ValueError, match='time-invariant'):
        model(wrong, cluster='region')
    wrong=base.copy()
    wrong.loc[(wrong.unit==0), 'region']=None
    with pytest.raises(ValueError, match='missing'):
        model(wrong, cluster='region')
    wrong=base.copy()
    wrong['region'] = wrong['unit'] // 24  # only 7 clusters
    with pytest.raises(ValueError, match='at least 10'):
        model(wrong, cluster='region')


def test_cluster_cells_require_independent_controls_and_treated_clusters():
    wrong=clustered_panel()
    # Most donor units sit in just one independent control cluster while
    # there are >=10 clusters overall, so nominal unit counts are misleading.
    wrong.loc[wrong.cohort == -1, 'region'] = 'one_donor_cluster'
    with pytest.raises(ValueError, match='No identified post-treatment'):
        model(wrong, cluster='region')


def test_invalid_cluster_scores_rejected():
    with pytest.raises(ValueError, match='at least 10'):
        InferencePlan.build(np.zeros(100, dtype=int), 100)
    plan = InferencePlan.build(np.repeat(np.arange(10), 10), 100)
    with pytest.raises(ValueError, match='one row'):
        plan.se(np.ones(10))
    with pytest.raises(ValueError, match='Nonfinite'):
        plan.se(np.full(100,np.nan))


def test_clustered_traditional_dr_with_nuisance_covariates_preserves_point():
    from scipy.special import expit
    rng=np.random.default_rng(249)
    rows=[]
    # Multiple group cohorts inside each independent region. Treatment is
    # correlated with covariates and serial shocks are region-correlated.
    for region in range(30):
        post_shock=rng.normal(0,.5)
        for j in range(12):
            unit=region*12+j
            x=rng.normal()
            treated=bool(rng.binomial(1,expit(.2+.35*x)))
            baseline=rng.normal()
            for t in range(3):
                outcome=baseline+.6*t*x+((post_shock+rng.normal(0,.1)+1.1*treated) if t==2 else 0)
                rows.append({'unit':unit,'year':t,'cohort':2 if treated else -1,
                             'outcome':outcome,'region':region,'x':x})
    data=pd.DataFrame(rows)
    plain=model(data, covariates=['x'],est_method='dr')
    clustered=model(data,covariates=['x'],est_method='dr',cluster='region')
    assert clustered.estimate == pytest.approx(plain.estimate,abs=1e-12)
    # Coarser clustering need not *always* increase SE: common region shocks
    # can cancel between treated and controls in the same region.
    assert clustered.se != pytest.approx(plain.se, rel=0.01)
    assert clustered.model_meta['est_method']=='dr'
    assert clustered.model_meta['n_clusters']==30
    assert clustered.inference_details['method']=='cluster_level_conditional_if_analytical_t_experimental'
    assert not clustered.to_record().inference.independently_validated
