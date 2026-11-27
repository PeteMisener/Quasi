"""Exploratory fuzzy RD comparison: linearized RBC vs prior corrected-jump ratio.

Not reference parity and not a statistical certification test. All 500
replications use independent synthetic data with true local causal effect 1.8.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from quasi.rdd.local_inference import boundary_weights, rd_inference


def run(*, seed: int = 991, repetitions: int = 500, n: int = 850) -> dict:
    rng = np.random.default_rng(seed)
    truth = 1.8
    rows = []
    failures = 0
    for i in range(repetitions):
        x = rng.uniform(-1.5, 1.5, n)
        prob = np.clip(.22 + .48*(x>=0) + .1*x + .26*x*x*(x>=0), .05, .95)
        treatment = (rng.uniform(size=n) < prob).astype(float)
        y = 2 + truth*treatment + .3*x + 1.5*x*x*(x>=0) + rng.normal(0, .65, n)
        try:
            fit = rd_inference(x, y, .95, 'triangular', 1, treatment=treatment,
                               bias_bandwidth=1.3)
            L = (boundary_weights(x, .95, 1.3, 1, 'triangular', 'right').corrected
                 - boundary_weights(x, .95, 1.3, 1, 'triangular', 'left').corrected)
            old_ratio = float((L @ y) / (L @ treatment))
            rows.append((fit['estimate'], old_ratio, fit['conventional_estimate'], fit['se']))
        except (ValueError, FloatingPointError):
            failures += 1
    data = np.asarray(rows, dtype=float)
    if not data.size:
        raise ValueError('No successful simulations')
    cover = float(np.mean(np.abs(data[:, 0] - truth) <= 1.959963984540054 * data[:, 3]))
    delta = data[:, 0] - data[:, 1]
    return {
        'study': 'exploratory_fuzzy_rdd_corrected_ratio_vs_first_order',
        'certification_status': 'NOT_CERTIFIED',
        'independent_reference_execution': False,
        'simulation_seed': seed, 'requested_replications': repetitions,
        'successful_replications': len(data), 'failures': failures,
        'n_per_replication': n, 'true_cutoff_effect': truth,
        'h': .95, 'b': 1.3, 'degree': 1, 'kernel': 'triangular',
        'treatment_propensity': 'clip(.22+.48*I(x>=0)+.1*x+.26*x^2*I(x>=0),.05,.95)',
        'outcome': '2+1.8*d+.3*x+1.5*x^2*I(x>=0)+Normal(0,.65)',
        'mean_estimates': {
            'first_order_rbc': float(data[:, 0].mean()),
            'old_corrected_jump_ratio': float(data[:, 1].mean()),
            'conventional': float(data[:, 2].mean()),
        },
        'rmse': {
            'first_order_rbc': float(np.sqrt(np.mean((data[:, 0]-truth)**2))),
            'old_corrected_jump_ratio': float(np.sqrt(np.mean((data[:, 1]-truth)**2))),
            'conventional': float(np.sqrt(np.mean((data[:, 2]-truth)**2))),
        },
        'observed_95pct_coverage_first_order_rbc': cover,
        'coverage_mc_se': float(np.sqrt(cover*(1-cover)/len(data))),
        'delta_first_order_minus_old_percentiles': dict(zip(
            ('min','p05','median','p95','max'),
            map(float, np.percentile(delta, (0,5,50,95,100))))),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('phase7d_fuzzy_exploratory.json'))
    parser.add_argument('--repetitions', type=int, default=500)
    args = parser.parse_args()
    result = run(repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
