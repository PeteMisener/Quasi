"""Local Quasi values on the frozen RDD fixture for independent triage.

Never treat this file as an externally produced reference or parity result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import quasi
from validation.rdd_fixture_contract import load_phase7_fixture
from validation.rdd_reference_contract import PARITY_H, PARITY_B, PARITY_P, PARITY_Q, PARITY_VCE, PARITY_KERNEL


def candidate_values(fixture: Path) -> dict:
    data, fingerprint = load_phase7_fixture(fixture)
    rows = []
    for design in ('sharp', 'fuzzy'):
        for method in ('conventional', 'bias_corrected'):
            cls = quasi.FuzzyRDD if design == 'fuzzy' else quasi.SharpRDD
            obj = cls(data=data, outcome='y', running_var='x', cutoff=0.0,
                      treatment='d' if design == 'fuzzy' else None,
                      bandwidth=PARITY_H, inference=method,
                      bias_bandwidth=PARITY_B if method == 'bias_corrected' else None)
            result = obj.fit()
            rows.append({'design': design, 'method': method,
                'estimate': float(result.estimate), 'se': float(result.se),
                'inference_label': result.inference_details['method'],
                'n_effective': result.n_effective})
    return {
        'source': 'QUASI_INTERNAL_ONLY',
        'fixture_sha256': fingerprint,
        'settings': {'h': PARITY_H, 'b': PARITY_B, 'p': PARITY_P, 'q': PARITY_Q,
                     'vce': PARITY_VCE, 'kernel': PARITY_KERNEL, 'cutoff': 0.0},
        'external_rdrobust_parity': 'NOT_EXECUTED',
        'statistical_certification': 'NOT_CERTIFIED',
        'candidates': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, default=Path('validation/fixtures/rdd_phase7_fixed.csv'))
    parser.add_argument('--output', type=Path, default=Path('quasi_rdd_candidates.json'))
    args = parser.parse_args()
    candidate = candidate_values(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, indent=2) + '\n')
    print('Wrote Quasi-only candidate values (NOT external reference parity)')
