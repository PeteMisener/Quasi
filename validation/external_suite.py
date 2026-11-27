"""Execute and attest Quasi's *existing* external numerical reference workflows.

No installation, mock data, fixture substitution, or certification occurs here.
A missing independent runtime creates a BLOCKED row, not a passing test.
Every executed comparison must exit successfully AND write a validated evidence file.
Run from the repository root::

    python -m validation.external_suite --output validation/reports/external_suite.json
    python -m validation.external_suite --output validation/reports/external_suite.json --strict

The default runs available authentic references and reports unavailable references.
The strict mode intentionally exits nonzero until ALL reference comparisons run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'validation' / 'fixtures'

# Commands do not accept arbitrary file paths from JSON manifests or external data.
# The reference-generation commands are deliberately separate from Quasi comparisons.
REFERENCE_FIXTURES = {
    'iv_linearmodels': ('iv_phase8_synthetic.csv',),
    'rdd_python': ('rdd_phase7_fixed.csv',),
    'rdd_r': ('rdd_phase7_fixed.csv',),
    'did_r': (),  # did/DRDID package data are produced by independent R code.
    'sdid_r': ('sdid_phase10b_panel.csv', 'sdid_phase10c_draws.json'),
    'rdd_r_python_crosscheck': ('rdd_phase7_fixed.csv',),
}


def _digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def _version(name: str) -> str | None:
    try:
        if importlib.util.find_spec(name) is None:
            return None
        return importlib.metadata.version(name)
    except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError, ValueError):
        return None


def _r_requirements(names: tuple[str, ...]) -> tuple[bool, dict[str, str | None]]:
    exe = shutil.which('Rscript')
    if not exe:
        return False, {'Rscript': None, **{x: None for x in names}}
    # Static package names from trusted code, not user input.
    probe = (
        'cat(paste(sapply(c(' + ','.join('"' + x + '"' for x in names)
        + '),function(x) if(requireNamespace(x,quietly=TRUE)) as.character(packageVersion(x)) else "MISSING"),collapse="\\n"))'
    )
    try:
        proc = subprocess.run([exe, '-e', probe], cwd=ROOT, text=True,
                              capture_output=True, timeout=40, check=False)
        values = proc.stdout.strip().splitlines()
        versions = dict(zip(names, values)) if proc.returncode == 0 and len(values) == len(names) else {}
    except (OSError, subprocess.TimeoutExpired):
        versions = {}
    data = {'Rscript': exe, **{x: versions.get(x) for x in names}}
    return all(data.get(x) and data[x] != 'MISSING' for x in names), data


def _run_steps(commands: list[list[str]], outputs: list[Path], *, timeout: int) -> dict[str, Any]:
    logs = []
    for command in commands:
        try:
            cp = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, check=False,
                                env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + os.pathsep + str(ROOT)})
            logs.append({'argv': command, 'returncode': cp.returncode,
                         'stdout_tail': cp.stdout[-3000:], 'stderr_tail': cp.stderr[-3000:]})
            if cp.returncode != 0:
                return {'status': 'failed_execution', 'steps': logs, 'output_sha256': {}}
        except (OSError, subprocess.TimeoutExpired) as exc:
            logs.append({'argv': command, 'failure': type(exc).__name__})
            return {'status': 'failed_execution', 'steps': logs, 'output_sha256': {}}
    if any(not p.is_file() or p.stat().st_size == 0 for p in outputs):
        return {'status': 'failed_no_evidence', 'steps': logs, 'output_sha256': {}}
    return {'status': 'passed_numerical_unreviewed', 'steps': logs,
            'output_sha256': {str(p): _digest(p) for p in outputs}}


def _verify_comparison(file: Path, field: str, expected: object) -> None:
    value = json.loads(file.read_text(encoding='utf-8'))
    if value.get(field) != expected:
        raise ValueError(f'Invalid comparison contract: {file.name}.{field}')


def execute(output: Path) -> dict[str, Any]:
    """Run only present reference packages; never use fallback numerical outputs."""
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = output.parent / (output.stem + '_external_outputs')
    evidence.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    rscript = shutil.which('Rscript') or 'Rscript'
    availability = {'statsmodels': _version('statsmodels'),
                    'linearmodels': _version('linearmodels'),
                    'rdrobust': _version('rdrobust')}
    rdeps = {}
    for name, packages in [('did_r', ('did', 'DRDID')),
                           ('rdd_r', ('rdrobust', 'digest')),
                           ('sdid_r', ('synthdid', 'jsonlite'))]:
        available, versions = _r_requirements(packages)
        rdeps[name] = {'available': bool(available), 'versions': versions}
    provenance = {'python': sys.version.split()[0], 'packages': availability,
                  'r_references': rdeps,
                  'fixture_sha256': {name: _digest(FIXTURES / name)
                    for name in sorted({f for fs in REFERENCE_FIXTURES.values() for f in fs})}}

    def run(name: str, missing: list[str], commands: list[list[str]], files: list[Path],
            *, expected_contracts: list[tuple[Path, str, object]] = []):
        if missing:
            entries[name] = {'status': 'blocked_missing_dependency', 'missing': missing,
                             'steps': [], 'output_sha256': {}}
            return
        # Stale outputs must never count as current execution evidence.
        for f in files:
            f.unlink(missing_ok=True)
        row = _run_steps(commands, files, timeout=900)
        if row['status'] == 'passed_numerical_unreviewed':
            try:
                for file, key, expected in expected_contracts:
                    _verify_comparison(file, key, expected)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                row['status'] = 'failed_invalid_comparison'
                row['reason'] = str(exc)
        entries[name] = row

    def _has_statsmodels(ver: str | None) -> bool:
        if not ver:
            return False
        parts = tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in ver.split(".")[:3])
        return parts >= (0, 14, 6)

    entries: dict[str, dict[str, Any]] = {}
    stats = evidence / 'statsmodels_numerical.json'
    run('statsmodels_real_data', [] if _has_statsmodels(availability['statsmodels']) else ['statsmodels>=0.14.6'],
        [[py, '-m', 'validation.evidence_report', '--output', str(stats)]], [stats])
    if entries['statsmodels_real_data']['status'] == 'passed_numerical_unreviewed':
        obj = json.loads(stats.read_text())
        valid = [r for r in obj['references'] if r['id'] in ('grunfeld_2sls','nile_its_90','nile_its_95')]
        if len(valid) != 3 or any(r['status'] != 'passed_numerical_unreviewed' for r in valid):
            entries['statsmodels_real_data']['status'] = 'failed_invalid_comparison'

    observational = evidence / 'observational_statsmodels.json'
    run('observational_statsmodels', [] if _has_statsmodels(availability['statsmodels']) else ['statsmodels>=0.14.6'],
        [[py, '-m', 'validation.observational.reference_phase11b', '--output', str(observational)]],
        [observational], expected_contracts=[(observational, 'passed', True)])
    if entries['observational_statsmodels']['status'] == 'passed_numerical_unreviewed':
        checks = json.loads(observational.read_text()).get('comparisons', [])
        if len(checks) < 10 or not all(c.get('passed') is True for c in checks):
            entries['observational_statsmodels']['status'] = 'failed_invalid_comparison'

    iv = evidence / 'iv_linearmodels.json'
    run('iv_linearmodels', [] if availability['linearmodels'] else ['linearmodels'],
        [[py, '-m', 'validation.iv_linearmodels_parity', '--output', str(iv)]], [iv],
        expected_contracts=[(iv, 'validation_tier', 'numerical_only_unreviewed')])
    rdd_py = evidence / 'rdd_python.json'
    run('rdd_python', [] if availability['rdrobust'] else ['Python rdrobust'],
        [[py, '-m', 'validation.compare_python_rdrobust', '--output', str(rdd_py)]], [rdd_py],
        expected_contracts=[(rdd_py, 'comparison_status', 'numerically_passed_unreviewed')])
    rdd_ref = evidence / 'rdd_r_reference.csv'
    rdd_cmp = evidence / 'rdd_r_comparison.json'
    run('rdd_r', [] if rdeps['rdd_r']['available'] else ['R rdrobust and digest'],
        [[rscript, 'validation/r/phase7_rd_external_reference.R',
          'validation/fixtures/rdd_phase7_fixed.csv', str(rdd_ref)],
         [py, '-m', 'validation.compare_phase7_r', '--reference', str(rdd_ref), '--output', str(rdd_cmp)]],
        [rdd_ref, rdd_cmp], expected_contracts=[(rdd_cmp, 'status', 'numerically_passed_unreviewed')])
    cross = evidence / 'rdd_r_python_comparison.json'
    missing = [name for name in ('rdd_python', 'rdd_r') if entries[name]['status'] != 'passed_numerical_unreviewed']
    run('rdd_r_python_crosscheck', missing,
        [[py, '-m', 'validation.crosscheck_rdd_references', '--r-reference', str(rdd_ref),
          '--python-reference', str(rdd_py), '--output', str(cross)]], [cross])
    did_dir = evidence / 'did_r_generated'
    # Clear all R producer outputs, not only the top-level candidate report.
    # Otherwise an old versions.txt/CSV could masquerade as a fresh run.
    if did_dir.exists():
        shutil.rmtree(did_dir)
    did_dir.mkdir(exist_ok=True)
    did_report = evidence / 'did_r_comparison.json'
    run('did_r', [] if rdeps['did_r']['available'] else ['R did and DRDID'],
        [[rscript, 'validation/r/phase6_did_external_reference.R', str(did_dir)],
         [py, '-m', 'validation.compare_phase6_r', str(did_dir), '--report', str(did_report)]],
        [did_report], expected_contracts=[(did_report, 'candidate_parity_passed', True)])
    sdid_r = evidence / 'sdid_r_reference.json'
    sdid_cmp = evidence / 'sdid_r_comparison.json'
    sdid_placebo_r = evidence / 'sdid_r_placebos.json'
    sdid_placebo_cmp = evidence / 'sdid_r_placebo_comparison.json'
    run('sdid_r', [] if rdeps['sdid_r']['available'] else ['R synthdid and jsonlite'],
        [[rscript, 'validation/r/phase10b_sdid_reference.R', str(sdid_r)],
         [py, '-m', 'validation.sdid_phase10b_reference', str(sdid_r), '--output', str(sdid_cmp)],
         [rscript, 'validation/r/phase10c_sdid_fixed_draw_placebos.R', str(sdid_placebo_r)],
         [py, '-m', 'validation.sdid_phase10c_placebo_reference', str(sdid_placebo_r),
          '--output', str(sdid_placebo_cmp)]],
        [sdid_r, sdid_cmp, sdid_placebo_r, sdid_placebo_cmp],
        expected_contracts=[(sdid_cmp, 'status', 'numeric_candidate_pass'),
                            (sdid_placebo_cmp, 'status', 'numeric_candidate_pass')])

    # These methods still lack a configured, independently executed numerical
    # reference: no combination of the checks above can certify the portfolio.
    missing_method_references = ['scm', 'psm']
    configured_passed = all(x['status'] == 'passed_numerical_unreviewed' for x in entries.values())
    result = {'schema_version': 1, 'purpose': 'independent_numerical_execution_not_statistical_certification',
              'statistical_certification': 'NOT_CERTIFIED',
              'all_configured_checks_passed': configured_passed,
              'all_external_parity_passed': False,
              'method_references_not_configured': missing_method_references,
              'provenance': provenance, 'checks': entries}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description='Run authentic external reference comparisons, never simulate parity')
    p.add_argument('--output', type=Path, default=Path('validation/reports/external_suite.json'))
    p.add_argument('--strict', action='store_true', help='Exit nonzero unless all reference comparisons execute and pass')
    args = p.parse_args()
    result = execute(args.output)
    for name, row in result['checks'].items():
        print(name, row['status'], ', '.join(row.get('missing', [])))
    if any(row['status'].startswith('failed') for row in result['checks'].values()):
        raise SystemExit(1)
    if args.strict and not result['all_external_parity_passed']:
        raise SystemExit('BLOCKED: one or more independent references did not execute')


if __name__ == '__main__':
    main()
