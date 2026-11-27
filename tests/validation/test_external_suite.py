"""External evidence must remain separate from test counts and certifications."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from validation import external_suite as lab


def test_real_reference_execution_records_unreviewed_not_approved(tmp_path):
    report = lab.execute(tmp_path / 'external.json')
    assert report['statistical_certification'] == 'NOT_CERTIFIED'
    assert report['checks']['statsmodels_real_data']['status'] == 'passed_numerical_unreviewed'
    assert report['checks']['statsmodels_real_data']['steps'][0]['returncode'] == 0
    assert not report['all_external_parity_passed']
    data = json.loads((tmp_path / 'external.json').read_text())
    for path, digest in data['checks']['statsmodels_real_data']['output_sha256'].items():
        assert hashlib.sha256(lab.Path(path).read_bytes()).hexdigest() == digest
    assert data['provenance']['packages']['statsmodels'] is not None


def test_absent_packages_cannot_be_treated_as_executed(monkeypatch, tmp_path):
    monkeypatch.setattr(lab, '_version', lambda name: None)
    monkeypatch.setattr(lab, '_r_requirements', lambda names: (False, {'Rscript': None}))
    report = lab.execute(tmp_path / 'missing.json')
    assert all(row['status'] == 'blocked_missing_dependency' for row in report['checks'].values())
    assert all(not row['steps'] and not row['output_sha256'] for row in report['checks'].values())
    assert not report['all_external_parity_passed']


def test_failed_subprocess_never_generates_pass_even_with_stale_evidence(tmp_path):
    evidence = tmp_path / 'stale.json'
    evidence.write_text('{"comparison_status":"numerically_passed_unreviewed"}')
    row = lab._run_steps([[sys.executable, '-c', 'raise SystemExit(4)']], [evidence], timeout=10)
    assert row['status'] == 'failed_execution'
    assert not row['output_sha256']
    assert row['steps'][0]['returncode'] == 4


def test_missing_output_is_not_success(tmp_path):
    row = lab._run_steps([[sys.executable, '-c', 'pass']], [tmp_path/'never-generated.json'], timeout=10)
    assert row['status'] == 'failed_no_evidence'


def test_expected_status_and_schema_checked(tmp_path):
    file = tmp_path / 'reference.json'
    file.write_text('{"comparison_status":"FAILED_NUMERICAL_PARITY"}')
    with pytest.raises(ValueError, match='Invalid comparison contract'):
        lab._verify_comparison(file, 'comparison_status', 'numerically_passed_unreviewed')
    file.write_text('{"comparison_status":"numerically_passed_unreviewed"}')
    lab._verify_comparison(file, 'comparison_status', 'numerically_passed_unreviewed')
    file.write_text('{bad JSON')
    with pytest.raises(json.JSONDecodeError):
        lab._verify_comparison(file, 'comparison_status', 'numerically_passed_unreviewed')


def test_expected_r_requirements_missing_when_rscript_absent(monkeypatch):
    monkeypatch.setattr(lab.shutil, 'which', lambda exe: None)
    ready, versions = lab._r_requirements(('did', 'DRDID'))
    assert not ready and versions['Rscript'] is None
    assert versions['did'] is None and versions['DRDID'] is None


def test_cli_strict_blocks_missing_references(tmp_path):
    output = tmp_path / 'strict.json'
    process = subprocess.run([sys.executable, '-m', 'validation.external_suite',
                              '--output', str(output), '--strict'],
                             cwd=lab.ROOT, text=True, capture_output=True, timeout=100)
    assert process.returncode != 0
    assert output.is_file()
    document = json.loads(output.read_text())
    assert not document['all_external_parity_passed']
    assert 'BLOCKED' in process.stderr


def test_outputs_include_fixture_checksums(tmp_path):
    report = lab.execute(tmp_path / 'ledger.json')
    fixtures = report['provenance']['fixture_sha256']
    assert fixtures
    for name, sha in fixtures.items():
        assert lab._digest(lab.FIXTURES/name) == sha
        assert len(sha) == 64


def test_no_reuse_of_stale_success_artifact(monkeypatch, tmp_path):
    # Simulate a reference package that is claimed installed, but its comparison
    # exits nonzero. A file from a previous run cannot promote this attempt.
    original = lab._version
    def availability(name):
        if name == 'linearmodels':
            return 'fake-installed-version'
        return original(name)
    monkeypatch.setattr(lab, '_version', availability)
    destination = tmp_path / 'suite.json'
    leftover = tmp_path / 'suite_external_outputs' / 'iv_linearmodels.json'
    leftover.parent.mkdir()
    leftover.write_text('{"validation_tier":"numerical_only_unreviewed"}')
    report = lab.execute(destination)
    assert report['checks']['iv_linearmodels']['status'] == 'failed_execution'
    assert not leftover.exists()
    assert not report['all_external_parity_passed']


def test_execution_ledger_is_only_evidence_not_release_approval(tmp_path):
    from validation.release_readiness import audit
    external = lab.execute(tmp_path/'ledger.json')
    report = audit(numerical_states={}, execution_ledger=external)
    assert not report['stable_release_approved']
    assert report['blocker_count'] >= 37
    iv = next(row for row in report['methods'] if row['id'] == 'iv_2sls')
    assert iv['checks']['external_execution']['iv_linearmodels'] == 'blocked_missing_dependency'
    its = next(row for row in report['methods'] if row['id'] == 'its')
    assert its['checks']['external_execution']['statsmodels_real_data'] == 'passed_numerical_unreviewed'
    scm = next(row for row in report['methods'] if row['id'] == 'scm')
    assert scm['checks']['external_execution']['method_reference'] == 'not_configured'


def test_reference_artifact_mutation_is_detected(tmp_path):
    from validation.release_readiness import audit
    external = lab.execute(tmp_path / 'ledger.json')
    file = lab.Path(next(iter(external['checks']['statsmodels_real_data']['output_sha256'])))
    original = file.read_bytes()
    try:
        file.write_bytes(original + b'\n')
        report = audit(numerical_states={}, execution_ledger=external)
        assert report['external_execution_states']['statsmodels_real_data'] == 'failed_evidence_integrity'
        assert not report['stable_release_approved']
    finally:
        file.write_bytes(original)


def test_ledger_missing_digest_or_execution_steps_cannot_pass():
    from validation.release_readiness import validate_execution_ledger
    fake = {'schema_version': 1, 'statistical_certification': 'NOT_CERTIFIED',
            'checks': {'fake': {'status': 'passed_numerical_unreviewed', 'steps': [],
                                'output_sha256': {}}}}
    assert validate_execution_ledger(fake) == {'fake': 'failed_evidence_integrity'}
    fake['statistical_certification'] = 'CERTIFIED'
    with pytest.raises(ValueError, match='Unrecognized'):
        validate_execution_ledger(fake)


def test_ledger_is_included_in_readiness_markdown(tmp_path):
    from validation.release_readiness import audit, to_markdown
    external = lab.execute(tmp_path / 'ledger.json')
    text = to_markdown(audit(numerical_states={}, execution_ledger=external))
    assert 'Executed reference ledger' in text
    assert 'blocked_missing_dependency' in text
    assert 'RELEASE BLOCKED' in text
