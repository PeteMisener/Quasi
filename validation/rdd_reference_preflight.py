"""Machine-readable Phase 7 availability report; *never* an approval gate.

Use --strict in CI to demand that both reference runtimes are installed.
No substitute for actually executing numerical comparisons.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version

from validation.rdd_fixture_contract import load_phase7_fixture


def preflight(fixture: Path) -> dict:
    _, fingerprint = load_phase7_fixture(fixture)
    try:
        package = version('rdrobust')
    except PackageNotFoundError:
        package = None
    python_package_importable = importlib.util.find_spec('rdrobust') is not None
    rscript = shutil.which('Rscript')
    r_packages_verified = False
    if rscript:
        # Check runtime packages, rather than treating Rscript alone as ready.
        try:
            probe = subprocess.run(
                [rscript, '-e', "quit(status=if (requireNamespace('rdrobust',quietly=TRUE) && requireNamespace('digest',quietly=TRUE)) 0 else 2)"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            r_packages_verified = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            r_packages_verified = False
    missing = []
    if package is None or not python_package_importable:
        missing.append('Python rdrobust distribution')
    if not rscript:
        missing.append('Rscript')
    elif not r_packages_verified:
        missing.append('R rdrobust and digest packages')
    return {
        'fixture_sha256': fingerprint,
        'python_rdrobust_version': package,
        'rscript_available': bool(rscript),
        'reference_runtime_status': 'READY_FOR_EXECUTION' if not missing else 'BLOCKED_MISSING_RUNTIME',
        'missing_requirements': missing,
        'r_packages_verified': r_packages_verified,
        'reference_numerical_parity': 'NOT_EXECUTED',
        'statistical_certification': 'NOT_CERTIFIED',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, default=Path('validation/fixtures/rdd_phase7_fixed.csv'))
    parser.add_argument('--output', type=Path, default=Path('rdd_reference_preflight.json'))
    parser.add_argument('--strict', action='store_true', help='Exit nonzero when runtime dependencies missing')
    args = parser.parse_args()
    result = preflight(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['reference_runtime_status'])
    if args.strict and result['reference_runtime_status'] != 'READY_FOR_EXECUTION':
        raise SystemExit(2)
