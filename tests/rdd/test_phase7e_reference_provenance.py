"""Phase 7E checks compare harness safety; synthetic reference values here are TEST DOUBLES.

These tests cannot and do not demonstrate actual R/Python rdrobust agreement.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from validation.rdd_fixture_contract import PHASE7_RDD_SHA256
from validation.rdd_reference_contract import REFERENCE_FIELDS, validate_r_reference
from validation.crosscheck_rdd_references import crosscheck
from validation.rdd_reference_preflight import preflight

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'validation/fixtures/rdd_phase7_fixed.csv'


def make_frame():
    # Deliberately constructed values are NOT third-party benchmarks.
    rows = []
    for d in ['sharp','fuzzy']:
        for m in ['conventional','bias_corrected']:
            rows.append(dict(design=d, method=m, estimate=1.0 if d == 'sharp' else 2.0,
                se=0.2 if m == 'conventional' else 0.25, reference_package='rdrobust',
                reference_version='1.3.0', fixture_sha256=PHASE7_RDD_SHA256, h=1.15,
                b=1.35, p=1, q=2, vce='hc0', kernel='tri', cutoff=0,
                generator='validation/r/phase7_rd_external_reference.R'))
    return pd.DataFrame(rows, columns=REFERENCE_FIELDS)


def make_python_payload():
    f = make_frame()
    return {'reference_package':'rdrobust','reference_version':'1.3.0',
        'fixture_sha256':PHASE7_RDD_SHA256,
        'parameters': {'h':1.15,'b':1.35,'p':1,'q':2,'vce':'hc0','kernel':'tri','cutoff':0.0},
        'reference_values': [{'design': r.design, 'method':r.method,
             'estimate':float(r.estimate), 'se':float(r.se)} for r in f.itertuples()]}


def test_strict_r_reference_schema_accepts_complete_test_double():
    assert len(validate_r_reference(make_frame(), PHASE7_RDD_SHA256)) == 4


@pytest.mark.parametrize('field,value', [
    ('fixture_sha256', '0'*64), ('h', 1.5), ('b', 1.5), ('p', 2), ('q', 3),
    ('vce','nn'), ('kernel','uni'), ('generator','unknown'),
    ('reference_version','unknown'), ('se',-1.0), ('estimate',float('nan')),
])
def test_r_provenance_fails_closed_for_altered_parameter(field,value):
    frame=make_frame()
    frame.loc[0,field]=value
    with pytest.raises(ValueError):
        validate_r_reference(frame, PHASE7_RDD_SHA256)


def test_r_reference_rejects_missing_and_duplicate_cells():
    ref=make_frame()
    with pytest.raises(ValueError, match='exactly one'):
        validate_r_reference(pd.concat([ref.iloc[:3],ref.iloc[:1]]), PHASE7_RDD_SHA256)
    with pytest.raises(ValueError, match='schema'):
        validate_r_reference(ref.drop(columns=['fixture_sha256']), PHASE7_RDD_SHA256)


def test_cross_language_reference_missing_files_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match='BLOCKED'):
        crosscheck(FIXTURE,tmp_path/'r.csv',tmp_path/'p.json',tmp_path/'out.json')


def test_cross_language_disagreement_fails_and_saves_diagnostic_evidence(tmp_path):
    r=tmp_path/'r.csv';p=tmp_path/'p.json';out=tmp_path/'out.json'
    make_frame().to_csv(r,index=False)
    py=make_python_payload()
    py['reference_values'][0]['estimate']+=0.5
    p.write_text(json.dumps(py))
    with pytest.raises(ValueError,match='mismatch'):
        crosscheck(FIXTURE,r,p,out)
    evidence=json.loads(out.read_text())
    assert evidence['status']=='FAILED_CROSS_LANGUAGE_PARITY'
    assert evidence['statistical_certification']=='NOT_CERTIFIED'
    assert not evidence['comparisons'][0]['point_pass']


def test_cross_language_contract_match_always_unreviewed(tmp_path):
    r=tmp_path/'r.csv';p=tmp_path/'p.json';out=tmp_path/'out.json'
    make_frame().to_csv(r,index=False)
    p.write_text(json.dumps(make_python_payload()))
    evidence=crosscheck(FIXTURE,r,p,out)
    assert evidence['status']=='numerically_passed_unreviewed'
    assert evidence['statistical_certification']=='NOT_CERTIFIED'


def test_cross_language_rejects_modified_inputs_and_duplicate_python_cells(tmp_path):
    r=tmp_path/'r.csv';p=tmp_path/'p.json';out=tmp_path/'out.json'
    make_frame().to_csv(r,index=False)
    payload=make_python_payload()
    payload['fixture_sha256']='0'*64
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='fixture'):
        crosscheck(FIXTURE,r,p,out)
    payload=make_python_payload()
    payload['reference_values'][3]['method']='conventional'
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Duplicate'):
        crosscheck(FIXTURE,r,p,out)
    assert not out.exists()


def test_preflight_is_explicitly_not_certification():
    result=preflight(FIXTURE)
    assert result['fixture_sha256']==PHASE7_RDD_SHA256
    assert result['reference_numerical_parity']=='NOT_EXECUTED'
    assert result['statistical_certification']=='NOT_CERTIFIED'
    assert result['reference_runtime_status'] in ('READY_FOR_EXECUTION','BLOCKED_MISSING_RUNTIME')


def test_quasi_local_candidates_are_never_labeled_external():
    from validation.rdd_quasi_candidates import candidate_values
    candidate = candidate_values(FIXTURE)
    assert candidate['source']=='QUASI_INTERNAL_ONLY'
    assert candidate['external_rdrobust_parity']=='NOT_EXECUTED'
    assert candidate['statistical_certification']=='NOT_CERTIFIED'
    assert len(candidate['candidates'])==4


def test_reference_workflows_use_importable_module_invocation():
    workflows = ROOT / '.github/workflows'
    body = (workflows / 'external-parity-suite.yml').read_text()
    for required in (
        '-m validation.compare_python_rdrobust',
        '-m validation.compare_phase7_r',
        '-m validation.crosscheck_rdd_references',
    ):
        assert required in body
    assert 'python validation/' not in body
    r_generator = (ROOT / 'validation/r/phase7_rd_external_reference.R').read_text()
    assert 'digest::digest(file=args[[1]], algo=\'sha256\')' in r_generator
    assert PHASE7_RDD_SHA256 in r_generator
