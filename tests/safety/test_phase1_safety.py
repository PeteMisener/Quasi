"""Reproductions and containment checks for review issues Q-003–Q-007.

These tests establish *behavioral safeguards*, not statistical reference parity.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

import quasi
from quasi.base import DiagnosticStatus, DiagnosticTest, QEDResult


class DummyResult(QEDResult):
    def summary(self): return 'summary'
    def plot(self, **kwargs): raise NotImplementedError
    def bibtex(self): return 'citation'
    def methods_text(self): return 'methods'


def result_with_diagnostics(diags):
    return DummyResult(1, .1, .01, (.8, 1.2), 'ATT', 'y', 'test', 10, 5, diagnostics=diags)


def test_no_diagnostics_does_not_certify_identification():
    r = result_with_diagnostics({})
    assert not r.passed_validation
    assert r.identification_status == 'not_established'
    assert 'VALID IDENTIFICATION' not in r._repr_html_()
    assert 'identification_status' in r.to_metadata_dict()
    assert 'caused' not in r.explain(audience='stakeholder').lower()


def test_warn_does_not_count_as_all_diagnostics_passed():
    t = DiagnosticTest('test', DiagnosticStatus.WARN, 'z', 1.0, None, 'n/a', 'inconclusive')
    r = result_with_diagnostics({'t': t})
    assert not r.passed_validation
    assert 'VALID IDENTIFICATION' not in r._repr_html_()


def did_panel(first_cohorts=(0, 2, -1), n_per=10):
    rng = np.random.default_rng(18)
    rows = []
    for i, g in enumerate([g for g in first_cohorts for _ in range(n_per)]):
        for t in range(4):
            d = int(g >= 0 and t >= g)
            rows.append((i, t, d, 2 + 0.2*t + 2*d + rng.normal(0, .05)))
    return pd.DataFrame(rows, columns=['id','t','d','y'])


def test_did_early_cohort_explicitly_rejected():
    df = did_panel()
    with pytest.raises(ValueError, match='first observed period|no pre-treatment'):
        quasi.StaggeredDiD(df, 'y', 'id', 't', treatment='d').fit()


def test_did_reversible_treatment_rejected():
    df = did_panel((2, -1))
    df.loc[(df.id == 0) & (df.t == 3), 'd'] = 0
    with pytest.raises(ValueError, match='revers|absorbing'):
        quasi.StaggeredDiD(df, 'y', 'id', 't', treatment='d').fit()


def test_did_missing_unit_period_rejected():
    df = did_panel((2, -1))
    df = df[~((df.id == 0) & (df.t == 1))]
    with pytest.raises(ValueError, match='balanced|missing'):
        quasi.StaggeredDiD(df, 'y', 'id', 't', treatment='d').fit()


def test_did_options_not_silently_ignored():
    df = did_panel((2, -1))
    df['w'] = np.arange(len(df))
    with pytest.raises(NotImplementedError, match="covariates"):
        quasi.StaggeredDiD(df, "y", "id", "t", treatment="d", covariates=["w"])
    # The previously unsupported cluster argument must now be retained
    # and reflected in inference, not silently ignored.
    clustered = quasi.StaggeredDiD(df, "y", "id", "t", treatment="d", cluster="id", b_reps=19).fit()
    assert clustered.model_meta["cluster"] == "id"
    assert clustered.inference_details["se_unit"] == "cluster"


def iv_data():
    rng = np.random.default_rng(33)
    n = 100
    z, noise = rng.normal(size=(2,n))
    x = z + noise * .2
    return pd.DataFrame({'y':2*x+noise, 'x':x, 'z':z, 'grp':np.repeat(np.arange(20),5)})


def test_iv_cluster_requires_many_independent_clusters():
    df = iv_data()
    df['grp'] = np.repeat(np.arange(5), 20)
    with pytest.raises(ValueError, match='at least 10 distinct clusters'):
        quasi.TwoStageLS(df, 'y', 'x', ['z'], cluster='grp').fit()


def test_iv_plot_runs():
    import matplotlib.pyplot as plt
    fig, ax = quasi.TwoStageLS(iv_data(), 'y', 'x', ['z']).fit().plot()
    assert fig is not None and ax is not None
    plt.close(fig)


def its_data():
    rng = np.random.default_rng(70)
    t = np.arange(50)
    return pd.DataFrame({'t':t,'y':1+.1*t+3*(t>=24)+rng.normal(0,.4,len(t))})


def test_its_alpha_changes_ci():
    df = its_data()
    a = quasi.InterruptedTimeSeries(df, 'y','t',24,alpha=.05).fit()
    b = quasi.InterruptedTimeSeries(df, 'y','t',24,alpha=.10).fit()
    assert a.ci_lower < b.ci_lower < b.ci_upper < a.ci_upper


def test_its_duplicate_times_rejected():
    df = its_data()
    df.loc[10,'t'] = 9
    with pytest.raises(ValueError, match='duplicate|unique'):
        quasi.InterruptedTimeSeries(df, 'y','t',24).fit()


def synth_data():
    rng = np.random.default_rng(55)
    return pd.DataFrame([
        (unit,t, 5+t*0.5+rng.normal(0,.1) + (5 if unit == 'trt' and t >= 4 else 0), 1.0)
        for unit in ['trt','c1','c2','c3','c4'] for t in range(7)
    ], columns=['unit','t','y','w'])


def test_sdid_covariates_not_silently_ignored():
    with pytest.raises(NotImplementedError, match='covariates'):
        quasi.SyntheticDiD(synth_data(), 'y','unit','t','trt',4,covariates=['w'])


def test_sdid_placebo_excludes_real_treated_donor(monkeypatch):
    from quasi.synth import sdid
    real = sdid._sdid_weights
    inputs = []
    def spy(treated, donors, split, **kwargs):
        inputs.append((np.asarray(treated).copy(), np.asarray(donors).copy()))
        return real(treated, donors, split, **kwargs)
    monkeypatch.setattr(sdid, '_sdid_weights', spy)
    r = quasi.SyntheticDiD(synth_data(), 'y', 'unit', 't', 'trt', 4, n_placebos=1).fit()
    assert r.placebo_p_val is not None
    # The actual treated trajectory never enters *any* placebo donor pool.
    original_treated = synth_data().query('unit == "trt"').sort_values('t').y.to_numpy()
    assert all(not any(np.array_equal(row, original_treated) for row in donors)
               for _, donors in inputs[1:])
    assert 'exact randomization p-value' not in r.methods_text().lower()


def test_synthetic_control_not_alias_of_sdid():
    assert quasi.SyntheticControl is not quasi.SyntheticDiD
    fitted = quasi.SyntheticControl(synth_data(), 'y','unit','t','trt',4).fit()
    assert fitted.model_name == 'SyntheticControl'
    assert fitted.inference_details['method'] == 'not_available'


def test_sharp_and_fuzzy_are_distinct_and_validated():
    assert quasi.SharpRDD is not quasi.FuzzyRDD
    data = pd.DataFrame({'x':[-2,-1,0,1], 'y':[1,2,3,4], 'd':[0,0,1,1]})
    with pytest.raises(ValueError, match='treatment'):
        quasi.SharpRDD(data, 'y', running_var='x',cutoff=0, treatment='d')
    with pytest.raises(ValueError, match='treatment'):
        quasi.FuzzyRDD(data, 'y',running_var='x',cutoff=0)


def test_rdd_adjustment_covariates_rejected():
    data = pd.DataFrame({'x':[-2,-1,0,1], 'y':[1,2,3,4], 'w':[0,0,1,1]})
    with pytest.raises(NotImplementedError, match='covariates'):
        quasi.Discontinuity(data, 'y', running_var='x',cutoff=0,covariates=['w'])


def test_router_does_not_silently_drop_cluster_for_its():
    df = its_data()
    df['cluster'] = 1
    with pytest.raises(NotImplementedError, match='cluster'):
        quasi.estimate(df, 'y', time='t', intervention_date=24, cluster='cluster')


def test_custom_ci_level_export_and_html():
    r = quasi.InterruptedTimeSeries(its_data(), 'y', 't', 24, alpha=.10).fit()
    pm = r.to_metadata_dict()['primary_metric']
    assert pm['confidence_level'] == pytest.approx(.90)
    assert pm['ci'] == (r.ci_lower, r.ci_upper)
    assert 'ci_95' not in pm
    assert '90% Conf. Int.' in r._repr_html_()
    assert 'EXPERIMENTAL' in r._repr_html_()
    assert '90% Level CI' in r.summary()


def test_sdid_missing_donor_period_rejected():
    df = synth_data()
    df = df[~((df.unit == 'c1') & (df.t == 2))]
    with pytest.raises(ValueError, match='balanced|missing'):
        quasi.SyntheticDiD(df, 'y', 'unit','t','trt',4,n_placebos=0).fit()


def test_did_inconsistent_first_treated_indicator_rejected():
    df = did_panel((2,-1))
    df['first'] = df.groupby('id')['d'].transform(lambda s: 2 if s.max() else -1)
    df.loc[(df.id == 0) & (df.t == 2),'d'] = 0
    with pytest.raises(ValueError, match='inconsistent'):
        quasi.StaggeredDiD(df,'y','id','t',treatment='d',first_treated='first').fit()


def test_model_comparison_respects_custom_interval():
    a = quasi.InterruptedTimeSeries(its_data(), 'y','t',24,alpha=.10).fit()
    b = quasi.InterruptedTimeSeries(its_data(), 'y','t',24,alpha=.05).fit()
    table = quasi.compare_models({'90%': a, '95%': b}, se_type='brackets')
    md = table.to_markdown()
    assert f'{a.ci_lower:.3f}' in md
    assert f'{b.ci_upper:.3f}' in md


def test_legacy_parity_cli_disabled_until_authentic_goldens():
    from tests.run_benchmarks import run_benchmark_suite
    with pytest.raises(RuntimeError, match='NOT R parity tests'):
        run_benchmark_suite()
