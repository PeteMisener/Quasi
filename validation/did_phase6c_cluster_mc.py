"""Seeded exploratory DiD cluster inference experiment; never certification.

Run with: PYTHONPATH=src:. python -m validation.did_phase6c_cluster_mc --reps 300
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from quasi import StaggeredDiD


def make_data(seed: int, effect: float, *, clusters: int = 24, units_per_cluster: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(clusters):
        treated = group < clusters//2
        # Independent clusters, correlated responses within a cluster.
        post_shock = rng.normal(scale=.9)
        for j in range(units_per_cluster):
            unit = group*units_per_cluster+j
            intercept = rng.normal()
            idiosyncratic = rng.normal(scale=.2)
            for t in range(3):
                y = intercept + .3*t + (post_shock + idiosyncratic + effect*treated if t == 2 else 0)
                rows.append({'unit':unit, 'year':t, 'cohort':2 if treated else -1,
                             'outcome':y, 'region':group})
    return pd.DataFrame(rows)


def run(reps: int = 300, start_seed: int = 1100) -> dict:
    if reps < 10:
        raise ValueError('Exploratory comparison needs at least 10 replications')
    output = {'qualification':'exploratory synthetic experiment, not independent method certification',
              'reps':reps, 'number_of_clusters':24, 'units_per_cluster':7,
              'clustered_variance':'IF cluster sums * M/(M-1); t_(M-1) pointwise critical'}
    for effect in (0.0, 1.25):
        stats = {m:{'covered':0,'rejected_null':0,'bias':[],'se':[],'failures':0} for m in ('unit','coarse_cluster')}
        for iteration in range(reps):
            data = make_data(start_seed+iteration, effect)
            for name in stats:
                try:
                    result = StaggeredDiD(data, 'outcome', 'unit', 'year', first_treated='cohort',
                                          cluster='region' if name=='coarse_cluster' else None,
                                          b_reps=39, seed=77).fit()
                except Exception:
                    stats[name]['failures'] += 1
                    continue
                rec=stats[name]
                rec['covered'] += result.ci_lower<=effect<=result.ci_upper
                rec['rejected_null'] += result.p_value < .05
                rec['bias'].append(result.estimate-effect)
                rec['se'].append(result.se)
        output[f'effect_{effect:g}']={}
        for name,stat in stats.items():
            valid=reps-stat['failures']
            output[f'effect_{effect:g}'][name]={
                'failures':stat['failures'], 'coverage':stat['covered']/valid if valid else None,
                'null_rejection_frequency':stat['rejected_null']/valid if valid else None,
                'mean_bias':float(np.mean(stat['bias'])) if valid else None,
                'mean_se':float(np.mean(stat['se'])) if valid else None,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=300)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(reps=args.reps)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__=='__main__':
    main()
