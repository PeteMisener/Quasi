"""Fail-closed independently executed Python rdrobust numerical comparison.

Prerequisite: pip install rdrobust. Never replaces a missing reference with a
hand-generated expectation. Matched h/b and HC0 are intentional; agreement on
this fixed sample is NOT general method or statistical certification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import numpy as np
import pandas as pd
import quasi
from validation.rdd_fixture_contract import load_phase7_fixture
from validation.rdd_reference_contract import (PARITY_H, PARITY_B, PARITY_P, PARITY_Q,
    PARITY_KERNEL, PARITY_VCE, PARITY_CUTOFF)


def compare(fixture: Path, output: Path, *, point_atol: float = 1e-5,
            se_rtol: float = 0.03) -> dict:
    # Verify the frozen fixture even when the optional reference package is missing.
    data, fixture_hash = load_phase7_fixture(fixture)
    try:
        from rdrobust import rdrobust
        ref_version = version('rdrobust')
    except (ImportError, PackageNotFoundError) as err:
        raise RuntimeError('External rdrobust unavailable; parity gate blocked') from err

    results = {'reference_package': 'rdrobust', 'reference_version': ref_version,
               'fixture_sha256': fixture_hash,
               'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'inference_status': 'NOT_CERTIFIED', 'comparison_status': 'pending',
               'tolerances': {'point_atol': point_atol, 'se_rtol': se_rtol},
               'parameters': {'h': PARITY_H, 'b': PARITY_B, 'p': PARITY_P, 'q': PARITY_Q,
                              'kernel': PARITY_KERNEL, 'vce': PARITY_VCE, 'cutoff': PARITY_CUTOFF},
               'reference_values': [], 'comparisons': []}
    all_ok = True
    if point_atol < 0 or not np.isfinite(point_atol) or se_rtol < 0 or not np.isfinite(se_rtol):
        raise ValueError('Invalid reference comparison tolerances')
    for design in ('sharp', 'fuzzy'):
        inputs = dict(y=data.y.to_numpy(), x=data.x.to_numpy(), c=0,
                      h=1.15, b=1.35, p=1, q=2, vce='hc0', kernel='tri',
                      masspoints='adjust')
        if design == 'fuzzy':
            inputs['fuzzy'] = data.d.to_numpy()
        external = rdrobust(**inputs)
        # Contract is the three conventional / bias-corrected / robust rows.
        # Robust inference pairs row 2 point estimate with row 3 SE.
        coef = getattr(external, 'coef', None)
        se = getattr(external, 'se', None)
        if not isinstance(coef, pd.DataFrame) or not isinstance(se, pd.DataFrame):
            raise ValueError('Unexpected rdrobust coefficient/SE layout')
        if coef.shape[0] != 3 or se.shape[0] != 3 or coef.shape[1] != 1 or se.shape[1] != 1:
            raise ValueError('Unexpected rdrobust coefficient/SE shape')
        expected = ['Conventional', 'Bias-Corrected', 'Robust']
        if [str(x).lower().replace(' ', '-') for x in coef.index] != [x.lower() for x in expected]:
            raise ValueError('Unexpected rdrobust coefficient row names')
        if [str(x).lower().replace(' ', '-') for x in se.index] != [x.lower() for x in expected]:
            raise ValueError('Unexpected rdrobust SE row names')
        ref_coefficients = coef.iloc[:, 0].to_numpy(dtype=float)
        ref_ses = se.iloc[:, 0].to_numpy(dtype=float)
        if not np.isfinite(ref_coefficients).all() or not np.isfinite(ref_ses).all():
            raise ValueError('Non-finite independent rdrobust reference results')
        for method, idx, se_idx in (('conventional', 0, 0), ('bias_corrected', 1, 2)):
            results['reference_values'].append({'design': design, 'method': method,
                'estimate': float(ref_coefficients[idx]), 'se': float(ref_ses[se_idx])})
        Model = quasi.FuzzyRDD if design == 'fuzzy' else quasi.SharpRDD
        for method, idx, se_idx in (('conventional', 0, 0), ('bias_corrected', 1, 2)):
            fit = Model(data, 'y', 'x', 0, treatment='d' if design=='fuzzy' else None,
                        bandwidth=1.15, inference=method,
                        bias_bandwidth=1.35 if method == 'bias_corrected' else None).fit()
            pass_point = bool(np.isclose(fit.estimate, ref_coefficients[idx], atol=point_atol, rtol=0))
            pass_se = bool(np.isclose(fit.se, ref_ses[se_idx], atol=1e-8, rtol=se_rtol))
            all_ok &= pass_point and pass_se
            results['comparisons'].append({'design': design, 'inference': method,
                 'quasi_estimate': fit.estimate, 'reference_estimate': float(ref_coefficients[idx]),
                 'quasi_se': fit.se, 'reference_se': float(ref_ses[se_idx]),
                 'point_pass': pass_point, 'se_pass': pass_se})
    results['comparison_status'] = 'numerically_passed_unreviewed' if all_ok else 'FAILED_NUMERICAL_PARITY'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n')
    if not all_ok:
        raise ValueError(f'Numerical parity failed; inspect evidence at {output}')
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, default=Path('validation/fixtures/rdd_phase7_fixed.csv'))
    parser.add_argument('--output', type=Path, default=Path('phase7_python_rdrobust.json'))
    a=parser.parse_args()
    result=compare(a.fixture,a.output)
    print(result['comparison_status'], '— not statistically certified')
