"""Release inventory must not imply that implemented means statistically approved."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from validation import release_readiness

ROOT = Path(__file__).resolve().parents[2]
NUMERICAL = {'grunfeld_2sls': 'passed_numerical_unreviewed',
             'nile_its_95': 'passed_numerical_unreviewed',
             'nile_its_90': 'passed_numerical_unreviewed'}


def test_all_ten_estimators_included_without_false_certification():
    report = release_readiness.audit(numerical_states=NUMERICAL)
    ids = {m['id'] for m in report['methods']}
    assert ids == {'staggered_did', 'sharp_rdd', 'fuzzy_rdd', 'iv_2sls', 'its',
                   'scm', 'sdid', 'aipw_att', 'ipw_att', 'psm'}
    assert report['method_count'] == 10
    assert not report['stable_release_approved']
    assert all(m['release_tier'] == 'experimental' and not m['research_approved']
               for m in report['methods'])
    assert all(m['checks']['public_api'] == 'passed_present' for m in report['methods'])
    assert report['blocker_count'] == 3 * 10 + len(release_readiness.PROJECT_REQUIREMENTS)


def test_missing_reference_is_an_added_blocker_and_never_approves():
    report = release_readiness.audit(numerical_states={})
    assert any('local numerical reference is missing' in blocker
               for method in report['methods'] for blocker in method['blockers'])
    assert not report['stable_release_approved']


def test_failed_reference_reported_as_release_failure():
    changed = dict(NUMERICAL, grunfeld_2sls='failed_provenance')
    report = release_readiness.audit(numerical_states=changed)
    assert report['local_reference_failures'] == {'grunfeld_2sls': 'failed_provenance'}
    iv = next(m for m in report['methods'] if m['id'] == 'iv_2sls')
    assert any('local numerical check failed' in issue for issue in iv['blockers'])


def test_missing_public_export_fails_closed(monkeypatch):
    first = release_readiness.REQUIREMENTS[0]
    altered = release_readiness.MethodRequirement(
        first.id, ('FakeMethodThatCannotExist',), first.external_reference,
        first.parity_requirement, first.inference_requirement, first.remaining_limit)
    monkeypatch.setattr(release_readiness, 'REQUIREMENTS', (altered,))
    report = release_readiness.audit(numerical_states=NUMERICAL)
    assert report['methods'][0]['checks']['public_api'] == 'failed_missing_export'
    assert any('FakeMethodThatCannotExist' in issue for issue in report['methods'][0]['blockers'])


def test_requirements_registry_has_unique_ids_exports_and_nonempty_actions():
    assert len({m.id for m in release_readiness.REQUIREMENTS}) == len(release_readiness.REQUIREMENTS)
    assert len({name for m in release_readiness.REQUIREMENTS for name in m.public_entry_points}) == sum(
        len(m.public_entry_points) for m in release_readiness.REQUIREMENTS)
    for item in release_readiness.REQUIREMENTS:
        assert all((item.id, *item.public_entry_points, item.external_reference,
                    item.parity_requirement, item.inference_requirement, item.remaining_limit))


def test_readiness_output_json_markdown_and_strict_exit(tmp_path):
    pytest.importorskip('statsmodels')
    target = tmp_path / 'status.json'
    command = [sys.executable, '-m', 'validation.release_readiness', '--output', str(target)]
    success = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    report = json.loads(target.read_text())
    assert not report['stable_release_approved']
    assert 'RELEASE BLOCKED' in target.with_suffix('.md').read_text()
    assert len(report['methods']) == 10
    strict = subprocess.run(command + ['--strict'], cwd=ROOT, capture_output=True, text=True)
    assert strict.returncode != 0
    assert 'BLOCKED' in strict.stderr


def test_current_documentation_is_generated_from_method_registry():
    report = release_readiness.audit(numerical_states=NUMERICAL)
    saved = (ROOT / 'docs' / 'release-readiness.md').read_text(encoding='utf-8')
    assert saved == release_readiness.to_markdown(report)
