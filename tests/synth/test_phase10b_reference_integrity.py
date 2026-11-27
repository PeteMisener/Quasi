"""Reference contract negative controls: no live R benchmark is implied."""
import json
import numpy as np
import pytest

from validation import sdid_phase10b_reference as ref
from quasi.synth.sdid import _sdid_effect, _sdid_weights


def reference_stub():
    """Only for exercising fail-closed metadata checks, never released as R evidence."""
    m,Y,k=ref.fixture_inputs()
    omega,lam,_ = _sdid_weights(Y[-1],Y[:-1],k)
    return {'method':'R_synthdid_estimate', 'reference_execution':'R_synthdid',
            'fixture_sha256':m['sha256'],'design':m['design'],
            'sparsify':'none','omega_intercept':True,'lambda_intercept':True,
            'package_version':'not-a-real-version-test-stub',
            'omega':omega.tolist(),'lambda':lam.tolist(),
            'estimate':_sdid_effect(Y[-1],Y[:-1],k,omega,lam)}


def test_fixture_fingerprint_matches_and_data_is_balanced():
    m,Y,k = ref.fixture_inputs()
    assert m['n_pre']==k==9
    assert Y.shape==(13,15)
    assert len(m['sha256'])==64


def test_missing_reference_is_hard_error(tmp_path):
    assert not (tmp_path/'missing.json').exists()
    with pytest.raises(FileNotFoundError):
        ref.compare(tmp_path/'missing.json')


@pytest.mark.parametrize('field,alter',[
    ('fixture_sha256','f'*64),
    ('omega_intercept',False),('lambda_intercept',False),
    ('sparsify','function'),('method','Quasi_self_reference'),
    ('reference_execution','not_executed'),('package_version',''),
    ('design','different'),
])
def test_reference_contract_rejects_mismatch(tmp_path,field,alter):
    obj=reference_stub();obj[field]=alter
    dest=tmp_path/'stub.json';dest.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        ref.compare(dest)


def test_reference_requires_internal_consistency(tmp_path):
    obj=reference_stub();obj['estimate']+=.1
    dest=tmp_path/'stub.json';dest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='own weights'):
        ref.compare(dest)


def test_reference_requires_valid_weights(tmp_path):
    obj=reference_stub();obj['omega']=[.1]*12
    dest=tmp_path/'stub.json';dest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='weights'):
        ref.compare(dest)


def test_reference_altered_fixture_fails(monkeypatch,tmp_path):
    old=ref.FILE
    altered=tmp_path/'altered.csv';altered.write_bytes(old.read_bytes()+b'\n')
    monkeypatch.setattr(ref,'FILE',altered)
    with pytest.raises(ValueError,match='checksum'):
        ref.fixture_inputs()
