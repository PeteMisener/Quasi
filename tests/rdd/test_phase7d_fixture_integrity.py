"""Provenance and fail-closed tests; no fabricated external parity fixtures."""
from pathlib import Path
import pytest

from validation.rdd_fixture_contract import PHASE7_RDD_SHA256, load_phase7_fixture
from validation.compare_python_rdrobust import compare as compare_python

FIXTURE = Path(__file__).resolve().parents[2] / 'validation/fixtures/rdd_phase7_fixed.csv'


def test_immutable_real_rdd_fixture_contract():
    data, fingerprint = load_phase7_fixture(FIXTURE)
    assert fingerprint == PHASE7_RDD_SHA256
    assert len(data) == 1800


def test_mismatched_real_rdd_fixture_is_not_accepted(tmp_path):
    altered = tmp_path / 'alternative.csv'
    altered.write_bytes(FIXTURE.read_bytes().replace(b'\n', b'\r\n', 1))
    with pytest.raises(ValueError, match='integrity failed'):
        load_phase7_fixture(altered)


def test_python_parity_refuses_altered_fixture_even_without_rdrobust(tmp_path):
    altered = tmp_path / 'fixture.csv'
    altered.write_bytes(FIXTURE.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='integrity failed'):
        compare_python(altered, tmp_path/'result.json')
    assert not (tmp_path/'result.json').exists()


def test_python_reference_comparator_rejects_disagreement_and_saves_evidence(tmp_path, monkeypatch):
    """Simulated bad reference checks the harness, NOT the estimator's parity."""
    import json
    import sys
    import types
    import numpy as np
    import pandas as pd
    import validation.compare_python_rdrobust as harness

    def deliberately_disagree(**kwargs):
        labels = ['Conventional', 'Bias-Corrected', 'Robust']
        return types.SimpleNamespace(
            coef=pd.DataFrame(np.zeros((3, 1)), index=labels, columns=['Coeff']),
            se=pd.DataFrame(np.ones((3, 1)), index=labels, columns=['Std. Err.']),
        )

    monkeypatch.setitem(sys.modules, 'rdrobust', types.SimpleNamespace(rdrobust=deliberately_disagree))
    monkeypatch.setattr(harness, 'version', lambda name: 'TEST_DOUBLE_NOT_AUTHENTIC')
    output = tmp_path / 'disagreement.json'
    with pytest.raises(ValueError, match='Numerical parity failed'):
        harness.compare(FIXTURE, output)
    evidence = json.loads(output.read_text())
    assert evidence['comparison_status'] == 'FAILED_NUMERICAL_PARITY'
    assert evidence['inference_status'] == 'NOT_CERTIFIED'
    assert len(evidence['comparisons']) == 4
    assert not all(row['point_pass'] for row in evidence['comparisons'])
