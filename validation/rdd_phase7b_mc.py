"""Exploratory fixed-bandwidth RDD coverage and donut sensitivity.

Not an independent reference or general certification. Intentionally includes
an invalid near-cutoff contamination DGP to demonstrate a sensitivity analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
from quasi.rdd.regression import SharpRDD


def run(reps: int = 200, seed: int = 20261009, n: int = 500) -> dict:
    if reps < 1 or n < 50:
        raise ValueError("reps >=1 and n>=50 required")
    rng = np.random.default_rng(seed)
    true = 1.5
    out = {key: {kind: [] for kind in ('baseline', 'donut')}
           for key in ('heteroskedastic_clean', 'local_contamination')}
    for _ in range(reps):
        x = rng.uniform(-1.5, 1.5, n)
        eps = rng.normal(size=n) * (0.25 + 0.65 * np.abs(x))
        baseline = 1.0 + .35*x + .15*x*x + true*(x>=0) + eps
        for scenario in out:
            y = baseline.copy()
            if scenario == 'local_contamination':
                # Selected outcomes violate the smooth potential-outcome condition.
                y[(x >= 0) & (x <= .15)] += 1.1
            df = pd.DataFrame({'x': x, 'y': y})
            for method, radius in [('baseline', 0.0), ('donut', .15)]:
                fit = SharpRDD(df, outcome='y', running_var='x', cutoff=0.,
                               bandwidth=1.0, donut_radius=radius).fit()
                out[scenario][method].append((fit.estimate, fit.se, fit.ci_lower <= true <= fit.ci_upper))
    result = dict(kind='exploratory_not_parity', reps=reps, n=n, seed=seed,
                  truth=true, bandwidth=1.0, kernel='triangular',
                  variance='conventional_hc0', donut_radius=.15, scenarios={})
    for scenario in out:
        result['scenarios'][scenario] = {}
        for method in out[scenario]:
            raw = np.asarray(out[scenario][method], dtype=float)
            rate = float(raw[:,2].mean())
            ci = stats.binomtest(int(raw[:,2].sum()), n=reps).proportion_ci(
                confidence_level=.95, method='wilson')
            result['scenarios'][scenario][method] = dict(
                bias=float(raw[:,0].mean()-true),
                rmse=float(np.sqrt(np.mean((raw[:,0]-true)**2))),
                mean_se=float(raw[:,1].mean()),
                coverage=rate,
                coverage_wilson_95=[float(ci.low),float(ci.high)],
                failures=0)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--output', type=Path, default=Path('rdd_phase7b_exploratory.json'))
    args = parser.parse_args()
    evidence = run(args.reps, args.seed, args.n)
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
