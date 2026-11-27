"""Fail-closed, machine-readable v1.0 research readiness inventory.

This is a *requirements inventory*, not a certificate issuer. Only externally
reviewed evidence may close requirements; a green local pytest run must not.
Do not record fictitious numerical benchmark results in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

# A stable release is intentionally NOT inferred from source availability,
# local unit-test counts, reference scripts, or synthetic Monte Carlo evidence.
@dataclass(frozen=True)
class MethodRequirement:
    id: str
    public_entry_points: tuple[str, ...]
    external_reference: str
    parity_requirement: str
    inference_requirement: str
    remaining_limit: str
    numeric_check: str | None = None


REQUIREMENTS = (
    MethodRequirement('staggered_did', ('StaggeredDiD',), 'R did / DRDID',
                      'Authentic cohort-time ATT and aggregation parity',
                      'Cluster-aware coverage, influence functions, simultaneous bands',
                      'Conditional design and treatment-timing restrictions'),
    MethodRequirement('sharp_rdd', ('SharpRDD',), 'R/Python rdrobust',
                      'Fixed-bandwidth and selected-bandwidth point/SE/CI parity',
                      'Robust bias-corrected coverage and discrete-score behavior',
                      'Provisional density diagnostics and heuristic legacy bandwidth'),
    MethodRequirement('fuzzy_rdd', ('FuzzyRDD',), 'R/Python rdrobust',
                      'Wald jump, first-stage, bias-correction, and SE parity',
                      'Weak-first-stage and challenging-design coverage',
                      'Inference and first-stage uncertainty remain experimental'),
    MethodRequirement('iv_2sls', ('TwoStageLS',), 'Python linearmodels / Statsmodels',
                      '2SLS, robust/cluster covariance, and GMM J comparisons',
                      'Weak-IV confidence-set and AR inference calibration',
                      'Single-endogenous design and unavailable multiway clustering',
                      numeric_check='grunfeld_2sls'),
    MethodRequirement('its', ('InterruptedTimeSeries',), 'Statsmodels OLS/GLS/SARIMAX',
                      'HAC and AR(1) conditional numerical comparisons',
                      'Nominal coverage across serial dependence and finite samples',
                      'Observed undercoverage under strong AR(1) and misspecified AR(2)',
                      numeric_check='nile_its_95'),
    MethodRequirement('scm', ('SyntheticControl',), 'R Synth / validated SCM reference',
                      'Donor weights, predictor support, and point-effect parity',
                      'Design-appropriate placebo inference and uncertainty review',
                      'Outcome-only donor matching and unsupported primary SE'),
    MethodRequirement('sdid', ('SyntheticDiD',), 'R synthdid',
                      'Unit/time weights, regularization, and ATT parity',
                      'Validated placebo/bootstrap uncertainty by treated-unit regime',
                      'Uncertified primary SE/CI intentionally unavailable'),
    MethodRequirement('aipw_att', ('AIPWATT',), 'Independent DR ATT implementation',
                      'Nuisance fits and ATT numerical parity across DGPs',
                      'Cross-fitted and joint nuisance uncertainty calibration',
                      'Observed undercoverage in some correct-nuisance simulations'),
    MethodRequirement('ipw_att', ('IPWATT',), 'Independent weighted ATT implementation',
                      'Odds-weighted ATT and overlap parity',
                      'Finite-sample and cross-fitted variance validation',
                      'Cross-fitted IPW primary uncertainty unavailable'),
    MethodRequirement('psm', ('PropensityScoreMatching',), 'Independent k-NN matching reference',
                      'ATT, tie policy, reuse, caliper, and balance parity',
                      'Matching-appropriate uncertainty / asymptotic review',
                      'Primary SE/CI intentionally unavailable'),
)

# An execution ledger is evidence, not approval. The mapping makes missing
# statistical comparisons visible on the estimator's own readiness record.
METHOD_EXTERNAL_CHECKS = {
    'staggered_did': ('did_r',),
    'sharp_rdd': ('rdd_python', 'rdd_r', 'rdd_r_python_crosscheck'),
    'fuzzy_rdd': ('rdd_python', 'rdd_r', 'rdd_r_python_crosscheck'),
    'iv_2sls': ('iv_linearmodels', 'statsmodels_real_data'),
    'its': ('statsmodels_real_data',),
    'scm': (),
    'sdid': ('sdid_r',),
    'aipw_att': ('observational_statsmodels',),
    'ipw_att': ('observational_statsmodels',),
    'psm': (),
}


def validate_execution_ledger(ledger: dict) -> dict[str, str]:
    """Check that a declared numerical run has an existing, hashed artifact."""
    if ledger.get('schema_version') != 1 or ledger.get('statistical_certification') != 'NOT_CERTIFIED':
        raise ValueError('Unrecognized external execution ledger contract')
    checks = ledger.get('checks')
    if not isinstance(checks, dict) or not checks:
        raise ValueError('External execution ledger contains no checks')
    statuses = {}
    for name, row in checks.items():
        state = row.get('status')
        if state == 'passed_numerical_unreviewed':
            artifacts = row.get('output_sha256')
            steps = row.get('steps')
            if not isinstance(artifacts, dict) or not artifacts or not steps or any(
                step.get('returncode') != 0 for step in steps):
                state = 'failed_evidence_integrity'
            else:
                for path, digest in artifacts.items():
                    artifact = Path(path)
                    if (not artifact.is_file() or artifact.is_symlink() or
                        hashlib.sha256(artifact.read_bytes()).hexdigest() != digest):
                        state = 'failed_evidence_integrity'
                        break
        elif state not in ('blocked_missing_dependency', 'failed_execution',
                           'failed_no_evidence', 'failed_invalid_comparison'):
            state = 'failed_evidence_integrity'
        statuses[name] = state
    return statuses


PROJECT_REQUIREMENTS = (
    ('owner_and_distribution', 'Owner approval and distribution-name registry ownership'),
    ('license_and_provenance', 'Independent software/data license and provenance review'),
    ('external_ci', 'Passing fresh-dependency Python/OS CI and reference workflows'),
    ('release_api', 'Maintainer-approved public API and breaking-change migration'),
    ('independent_statistical_review', 'Named independent statistical sign-off for each stable estimator'),
    ('security_and_docs', 'Security contact, scanning, release documentation and user guidance'),
    ('coverage_and_fuzzing', 'Method-specific test coverage, adverse-case and simulation gates'),
)


def _numerical_states() -> dict[str, str]:
    """Execute available real-data reference checks; never silently fallback."""
    from validation.evidence_report import evaluate

    return {row['id']: row['status'] for row in evaluate()['references']}


def audit(*, numerical_states: dict[str, str] | None = None,
          execution_ledger: dict | None = None) -> dict[str, Any]:
    """Evaluate requirements, including evidence failures, without certification.

    ``numerical_states`` exists for deterministic tests; production calls execute
    reference checks from the pinned independent dataset and package.
    """
    states = _numerical_states() if numerical_states is None else dict(numerical_states)
    import quasi
    external_states = validate_execution_ledger(execution_ledger) if execution_ledger is not None else None

    methods = []
    for spec in REQUIREMENTS:
        missing_exports = [name for name in spec.public_entry_points if not hasattr(quasi, name)]
        local_reference = states.get(spec.numeric_check, 'not_available') if spec.numeric_check else 'not_available'
        # Even a complete local numerical check does not confer method approval.
        checks = {
            'public_api': 'failed_missing_export' if missing_exports else 'passed_present',
            'independent_point_parity': 'blocked_not_approved',
            'sampling_inference': 'blocked_not_approved',
            'independent_review': 'blocked_not_approved',
            'local_numerical_check': local_reference,
        }
        if local_reference.startswith('failed'):
            checks['local_numerical_check'] = local_reference
        if external_states is not None:
            refs = METHOD_EXTERNAL_CHECKS[spec.id]
            checks['external_execution'] = ({name: external_states.get(name, 'not_executed')
                                             for name in refs} if refs else
                                            {'method_reference': 'not_configured'})
        blockers = [
            f'{spec.id}: external reference parity not independently approved',
            f'{spec.id}: sampling inference not independently approved',
            f'{spec.id}: independent statistician approval absent',
        ]
        if missing_exports:
            blockers.append(f"{spec.id}: missing public exports: {', '.join(missing_exports)}")
        if spec.numeric_check and local_reference == 'not_available':
            blockers.append(f'{spec.id}: required local numerical reference is missing')
        if local_reference.startswith('failed'):
            blockers.append(f'{spec.id}: local numerical check failed: {local_reference}')
        methods.append({
            'id': spec.id, 'release_tier': 'experimental',
            'public_entry_points': list(spec.public_entry_points),
            'external_reference': spec.external_reference,
            'parity_requirement': spec.parity_requirement,
            'inference_requirement': spec.inference_requirement,
            'known_limit': spec.remaining_limit,
            'checks': checks, 'blockers': blockers,
            'research_approved': False,
        })
    project = [{'id': id_, 'description': desc, 'status': 'pending',
                'blocker': f'project: {desc}'} for id_, desc in PROJECT_REQUIREMENTS]
    # Missing reference fixtures are a separate *actual* execution failure, not
    # a routine pending certification requirement.
    failures = {id_: status for id_, status in states.items() if status.startswith('failed')}
    return {
        'schema_version': 1,
        'distribution': 'quasi-qed',
        'version': metadata.version('quasi-qed') if _distribution_installed() else quasi.__version__,
        'stable_release_approved': False,
        'method_count': len(methods),
        'methods': methods,
        'project_requirements': project,
        'local_reference_failures': failures,
        'external_execution_states': external_states,
        'blocker_count': sum(len(m['blockers']) for m in methods) + len(project),
        'note': 'Readiness inventory is not statistical certification; pending checks cannot be passed by a green pytest run.',
    }


def _distribution_installed() -> bool:
    try:
        metadata.version('quasi-qed')
        return True
    except metadata.PackageNotFoundError:
        return False


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        '# Quasi v1.0 method-by-method release readiness', '',
        '**RELEASE BLOCKED. Every estimator listed below is experimental.**', '',
        f"Methods audited: **{report['method_count']}**. Outstanding requirements: **{report['blocker_count']}**.", '',
        '| Estimator | Status | Reference needed | Principal limitations |',
        '|---|---|---|---|',
    ]
    for method in report['methods']:
        limits = method['known_limit'].replace('|', '\\|')
        lines.append(f"| `{method['id']}` | Experimental, blocked | "
                     f"{method['external_reference']} | {limits} |")
    if report.get('external_execution_states') is not None:
        lines += ['', '## Executed reference ledger (numerical only)', '',
                  '| Reference | Execution status |', '|---|---|']
        lines += [f'| `{name}` | `{state}` |' for name, state in report['external_execution_states'].items()]
    lines += ['', '## Project-wide blockers', '']
    lines += [f"- {item['description']}" for item in report['project_requirements']]
    lines += ['', '## Evidence interpretation', '',
              'Local numerical equality and synthetic simulations establish neither statistical coverage '
              'nor identification. References must be executed and reviewed independently. '
              'These requirements remain open until signed off through an accountable release process.', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Fail-closed Quasi research release inventory')
    parser.add_argument('--output', type=Path, help='Write machine-readable JSON and matching Markdown report')
    parser.add_argument('--external-ledger', type=Path, help='Optional actual reference execution ledger (read-only evidence)')
    parser.add_argument('--strict', action='store_true', help='Block when any release requirement remains')
    args = parser.parse_args()
    ledger = json.loads(args.external_ledger.read_text(encoding='utf-8')) if args.external_ledger else None
    result = audit(execution_ledger=ledger)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        args.output.with_suffix('.md').write_text(to_markdown(result), encoding='utf-8')
    print(f"Methods: {result['method_count']}; open requirements: {result['blocker_count']}; "
          f"research release approved: {result['stable_release_approved']}")
    if result['local_reference_failures']:
        raise SystemExit('ERROR: failed local numerical reference checks')
    if result['external_execution_states'] and any(
            value.startswith('failed') for value in result['external_execution_states'].values()):
        raise SystemExit('ERROR: failed or tampered external numerical evidence')
    if args.strict and result['blocker_count']:
        raise SystemExit('BLOCKED: research release requirements incomplete')


if __name__ == '__main__':
    main()
