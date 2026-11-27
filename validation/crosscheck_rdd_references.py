"""Compare *independently produced* R and Python rdrobust reference outputs.

This is an additional consistency check; it cannot independently guarantee the
identity or authenticity of files supplied outside verified CI runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation.rdd_fixture_contract import load_phase7_fixture
from validation.rdd_reference_contract import (
    PARITY_H, PARITY_B, PARITY_P, PARITY_Q, PARITY_VCE,
    PARITY_KERNEL, PARITY_CUTOFF, PARITY_DESIGNS, PARITY_METHODS,
    validate_r_reference,
)


def crosscheck(fixture: Path, r_csv: Path, python_json: Path, output: Path,
               *, estimate_atol: float = 1e-7, se_rtol: float = 1e-4) -> dict:
    if not r_csv.is_file() or not python_json.is_file():
        raise FileNotFoundError('Both authentic reference outputs required; cross-language gate BLOCKED')
    _, sha = load_phase7_fixture(fixture)
    r_frame = validate_r_reference(pd.read_csv(r_csv), sha)
    py = json.loads(python_json.read_text())
    if py.get('reference_package') != 'rdrobust' or py.get('fixture_sha256') != sha:
        raise ValueError('Python reference provenance/fixture mismatch')
    params = {'h': PARITY_H, 'b': PARITY_B, 'p': PARITY_P, 'q': PARITY_Q,
              'vce': PARITY_VCE, 'kernel': PARITY_KERNEL, 'cutoff': PARITY_CUTOFF}
    if py.get('parameters') != params:
        raise ValueError('Python reference specification mismatch')
    if not isinstance(py.get('reference_version'), str) or not py['reference_version'].strip():
        raise ValueError('Python reference version missing')
    expected = {(d, m) for d in PARITY_DESIGNS for m in PARITY_METHODS}
    ref_values = py.get('reference_values')
    if not isinstance(ref_values, list) or len(ref_values) != 4:
        raise ValueError('Python reference result grid missing or invalid')
    by_key = {}
    for item in ref_values:
        if not isinstance(item, dict) or set(item) != {'design','method','estimate','se'}:
            raise ValueError('Malformed Python reference row')
        key = (item['design'], item['method'])
        if key in by_key:
            raise ValueError('Duplicate Python reference row')
        for field in ('estimate', 'se'):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError('Invalid Python reference numeric value')
        if item['se'] <= 0:
            raise ValueError('Invalid Python reference SE')
        by_key[key] = item
    if set(by_key) != expected:
        raise ValueError('Incomplete Python reference grid')
    if any(v < 0 or not np.isfinite(v) for v in (estimate_atol,se_rtol)):
        raise ValueError('Invalid cross-language tolerances')
    rows = []
    passed = True
    for r in r_frame.to_dict('records'):
        p = by_key[(r['design'], r['method'])]
        point_ok = bool(np.isclose(r['estimate'], p['estimate'], atol=estimate_atol, rtol=0))
        se_ok = bool(np.isclose(r['se'], p['se'], atol=1e-10, rtol=se_rtol))
        passed &= point_ok and se_ok
        rows.append({'design': r['design'], 'method': r['method'],
                     'r_estimate': float(r['estimate']), 'python_estimate': p['estimate'],
                     'r_se': float(r['se']), 'python_se': p['se'],
                     'point_pass': point_ok, 'se_pass': se_ok})
    record = {
        'fixture_sha256': sha, 'r_reference_version': r_frame['reference_version'].iloc[0],
        'python_reference_version': py['reference_version'],
        'status': 'numerically_passed_unreviewed' if passed else 'FAILED_CROSS_LANGUAGE_PARITY',
        'statistical_certification': 'NOT_CERTIFIED',
        'tolerances': {'estimate_atol': estimate_atol, 'se_rtol': se_rtol},
        'comparisons': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    if not passed:
        raise ValueError(f'R/Python rdrobust reference mismatch; inspect {output}')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, default=Path('validation/fixtures/rdd_phase7_fixed.csv'))
    parser.add_argument('--r-reference', type=Path, required=True)
    parser.add_argument('--python-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('rdd_cross_language_evidence.json'))
    args = parser.parse_args()
    print(crosscheck(args.fixture, args.r_reference, args.python_reference, args.output)['status'])
