"""Exploratory, seeded coverage and bias experiment for conditional DiD.

Do not use as independent R parity or as statistical certification.
Usage: PYTHONPATH=src:. python -m validation.did_phase6b_mc --reps 240 --output evidence.json
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from quasi.did.staggered import StaggeredDiD


def draw_panel(seed: int, n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = expit(-.2 + .6*x)
    group = rng.binomial(1, p, n).astype(bool)
    u = rng.normal(size=n)
    intercept = .5 * x
    rows=[]
    for t in range(4):
        y = 5 + intercept + .2*t + .7*x*t + u*.5*t + (2 * (t>=2))*group
        rows.append(pd.DataFrame({'unit':np.arange(n), 'time':t, 'outcome':y,
                                 'cohort':np.where(group,2,-1), 'x':x}))
    return pd.concat(rows, ignore_index=True)


def evaluate(reps: int=240, seed: int=20261009, n: int=180) -> dict:
    metrics={method: {'estimate':[], 'se':[], 'cover':[], 'failures':0} for method in ('unadjusted', 'dr')}
    for i in range(reps):
        data=draw_panel(seed+i, n)
        for method in metrics:
            try:
                r=StaggeredDiD(data, outcome='outcome', unit='unit', time='time',
                               first_treated='cohort', covariates=['x'] if method=='dr' else None,
                               est_method=method, b_reps=19).fit()
                metrics[method]['estimate'].append(r.estimate)
                metrics[method]['se'].append(r.se)
                metrics[method]['cover'].append(r.ci_lower <= 2 <= r.ci_upper)
            except (ValueError, np.linalg.LinAlgError):
                metrics[method]['failures']+=1
    return {'reps_requested':reps, 'n_units':n, 'seed':seed, 'target':2.,
            'validation_status':'synthetic_exploratory_not_certified',
            'results': {key:{'successes':len(val['cover']), 'failures':val['failures'],
                             'bias':float(np.mean(val['estimate'])-2) if val['estimate'] else None,
                             'mean_se':float(np.mean(val['se'])) if val['se'] else None,
                             'coverage_95':float(np.mean(val['cover'])) if val['cover'] else None}
                        for key,val in metrics.items()}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=240)
    parser.add_argument('--n',type=int,default=180)
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    result=evaluate(reps=args.reps,n=args.n)
    out=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    if args.output:
        with open(args.output,'w') as f: f.write(out+'\n')
    print(out)
if __name__=='__main__':main()
