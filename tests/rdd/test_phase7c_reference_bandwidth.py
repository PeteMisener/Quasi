"""Reference bandwidth interface: delegation, fail-closed behavior and result provenance.

The mock tests exercise the adapter contract; they do NOT validate rdrobust's
mathematics or certify Quasi's inference. Real-package parity is a separate gate.
"""
import sys
import types
import numpy as np
import pandas as pd
import pytest

import quasi
from quasi.rdd import reference_bandwidth as adapter


@pytest.fixture
def data():
    rng = np.random.default_rng(1107)
    x = rng.uniform(-2, 2, 1500)
    y = 1.5 * (x >= 0) + 0.6 * x + 0.3 * x*x + rng.normal(size=len(x)) * .4
    d = ((x >= 0) & (rng.uniform(size=len(x)) < 0.95)) | ((x < 0) & (rng.uniform(size=len(x)) < .05))
    return pd.DataFrame(dict(x=x, y=y, d=d.astype(int)))


def fake_reference(monkeypatch, *, bws=None, error=None):
    recorded = []
    def rdbwselect(**kwargs):
        recorded.append(kwargs)
        if error:
            raise error
        frame = bws if bws is not None else pd.DataFrame([[1.1, 1.1, 1.6, 1.6]],
             index=['mserd' if kwargs['bwselect'] == 'mserd' else 'cerrd'],
             columns=['h (left)','h (right)','b (left)','b (right)'])
        return types.SimpleNamespace(bws=frame)
    original_import = adapter.import_module
    monkeypatch.setattr(adapter, 'import_module', lambda name: types.SimpleNamespace(rdbwselect=rdbwselect) if name == 'rdrobust' else original_import(name))
    return recorded


@pytest.mark.parametrize('method,expected', [('rdrobust_mserd','mserd'),('rdrobust_cerrd','cerrd')])
def test_selector_delegates_and_records_provenance(data, monkeypatch, method, expected):
    calls = fake_reference(monkeypatch)
    result = quasi.SharpRDD(data,'y','x',0, bandwidth=method, inference='bias_corrected').fit()
    assert result.bandwidth == pytest.approx(1.1)
    assert result.inference_details['pilot_bandwidth'] == pytest.approx(1.6)
    assert result.inference_details['bandwidth_selection'] == expected + '_delegated_to_rdrobust'
    assert result.inference_details['reference_bias_bandwidth'] == pytest.approx(1.6)
    assert result.inference_details['bandwidth_reference_vce'] == 'hc0'
    assert result.inference_details['bandwidth_reference_masspoints'] == 'adjust'
    assert result.inference_details['reference_certified'] is False
    assert calls[0]['bwselect'] == expected
    assert calls[0]['vce'] == 'hc0'
    assert calls[0]['p'] == 1 and calls[0]['q'] == 2
    assert calls[0]['kernel'] == 'tri'
    assert calls[0]['masspoints'] == 'adjust'
    assert 'fuzzy' not in calls[0]


def test_fuzzy_passes_actual_treatment_and_cutoff(data, monkeypatch):
    calls = fake_reference(monkeypatch)
    result = quasi.FuzzyRDD(data, 'y', 'x', 0, treatment='d', bandwidth='rdrobust_mserd').fit()
    np.testing.assert_array_equal(calls[0]['fuzzy'], data.d.to_numpy())
    assert result.is_fuzzy


def test_design_api_preserves_reference_choice(data, monkeypatch):
    calls = fake_reference(monkeypatch)
    design = quasi.RDDDesign(outcome='y', running_var='x', cutoff=0,
                              bandwidth='rdrobust_mserd', inference='bias_corrected')
    result = quasi.estimate(data, design=design)
    assert result.bandwidth == pytest.approx(1.1)
    assert calls


def test_explicit_bias_bandwidth_overrides_reference_bias_only(data, monkeypatch):
    fake_reference(monkeypatch)
    res = quasi.SharpRDD(data, 'y','x', 0, bandwidth='rdrobust_mserd',
                         inference='bias_corrected', bias_bandwidth=1.9).fit()
    assert res.inference_details['pilot_bandwidth'] == 1.9
    assert res.inference_details['reference_bias_bandwidth'] == 1.6
    assert res.bandwidth == 1.1


def test_missing_package_fails_closed(data, monkeypatch):
    def missing(name):
        raise ImportError('mock: no rdrobust')
    monkeypatch.setattr(adapter, 'import_module', missing)
    with pytest.raises(ImportError, match='No heuristic fallback'):
        quasi.SharpRDD(data,'y','x',0, bandwidth='rdrobust_cerrd').fit()


@pytest.mark.parametrize('bad_data', [
    [[1.1,1.3,1.6,1.6]], [[1.1,1.1,0.0,1.6]],
    [[float('nan'),1.1,1.6,1.6]],
])
def test_invalid_reference_output_rejected(data,monkeypatch,bad_data):
    frame=pd.DataFrame(bad_data, index=['mserd'],
        columns=['h (left)','h (right)','b (left)','b (right)'])
    fake_reference(monkeypatch,bws=frame)
    with pytest.raises(ValueError):
        quasi.SharpRDD(data, 'y','x',0,bandwidth='rdrobust_mserd').fit()


def test_reference_package_errors_not_masked(data, monkeypatch):
    fake_reference(monkeypatch,error=RuntimeError('mass point degeneracy'))
    with pytest.raises(RuntimeError, match='mass point'):
        quasi.SharpRDD(data,'y','x',0,bandwidth='rdrobust_mserd').fit()


def test_no_unknown_selector_and_no_bool_bandwidth(data):
    for bw in ['rdrobust_msetwo','unknown',True]:
        with pytest.raises(ValueError):
            quasi.SharpRDD(data,'y','x',0,bandwidth=bw)
        with pytest.raises(quasi.InvalidDesignError):
            quasi.RDDDesign(outcome='y',running_var='x',cutoff=0,bandwidth=bw)


def test_donut_failure_after_selected_bandwidth(data, monkeypatch):
    fake_reference(monkeypatch)
    with pytest.raises(ValueError, match='donut_radius must be smaller'):
        quasi.SharpRDD(data,'y','x',0,bandwidth='rdrobust_mserd',donut_radius=1.2).fit()


def test_python_reference_gate_blocks_missing_actual_package(tmp_path):
    """Strict runner cannot synthesize a pass when rdrobust is not installed."""
    import importlib.util
    if importlib.util.find_spec('rdrobust') is not None:
        pytest.skip('Genuine rdrobust installed; run independent parity workflow instead')
    from validation.compare_python_rdrobust import compare
    from pathlib import Path
    evidence = tmp_path / 'unverified.json'
    with pytest.raises(RuntimeError, match='parity gate blocked'):
        compare(Path('validation/fixtures/rdd_phase7_fixed.csv'), evidence)
    assert not evidence.exists()
