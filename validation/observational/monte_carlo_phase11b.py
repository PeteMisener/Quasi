"""Iteration 23 exploratory coverage and matching-sensitivity laboratory.

Run: PYTHONPATH=src python -m validation.observational.monte_carlo_phase11b --reps 150
All simulated data have known constant ATT=2; observational assumptions are
imposed by the DGP, not established for empirical datasets. Failures reported.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from quasi.matching.psm import AIPWATT, IPWATT, PropensityScoreMatching
from quasi.errors import QuasiError


def run(reps: int = 150, seed: int = 2317, n: int = 450) -> dict:
    if reps < 1 or n < 50:
        raise ValueError('reps and n must be sufficient')
    result = {'design': 'independent Bernoulli treatment, correctly specified logit and untreated linear outcome',
              'known_ATT': 2.0, 'reps': reps, 'n': n, 'seed': seed,
              'experimental': True, 'statistical_certification': False, 'scenarios': {}}
    methods = ['aipw','ipw','psm_k1','psm_k3']
    for label, strength in [('moderate_overlap', .7), ('limited_overlap', 1.8)]:
        statistics = {name: {'bias': [], 'coverage': [], 'max_smd': [], 'failures':{}} for name in methods}
        for i in range(reps):
            rng=np.random.default_rng(seed + 100000*(label=='limited_overlap') + i)
            x=rng.normal(size=n)
            d=rng.binomial(1,expit(strength*x+.1))
            y=1+x*1.4+2*d+rng.normal(scale=.9,size=n)
            df=pd.DataFrame({'y':y,'d':d,'x':x})
            fits=[('aipw',AIPWATT,{}),('ipw',IPWATT,{}),
                  ('psm_k1',PropensityScoreMatching,{'n_neighbors':1}),
                  ('psm_k3',PropensityScoreMatching,{'n_neighbors':3})]
            for name,model,kw in fits:
                box=statistics[name]
                try:
                    r=model(df,'y','d',['x'],**kw).fit()
                    box['bias'].append(float(r.estimate-2.0))
                    box['max_smd'].append(float(max(r.smd_after.values())))
                    if np.isfinite(r.se):
                        box['coverage'].append(bool(r.ci_lower <= 2.0 <= r.ci_upper))
                except (QuasiError, ValueError, np.linalg.LinAlgError) as err:
                    reason=type(err).__name__
                    box['failures'][reason]=box['failures'].get(reason,0)+1
        summary={}
        for method,values in statistics.items():
            b=np.asarray(values['bias'])
            summary[method]={
                'successes':int(len(b)), 'failures':values['failures'],
                'mean_bias':float(b.mean()) if b.size else None,
                'rmse':float(np.sqrt(np.mean(b*b))) if b.size else None,
                'mean_max_abs_smd':float(np.mean(values['max_smd'])) if b.size else None,
                'observed_95pc_coverage': (float(np.mean(values['coverage'])) if values['coverage'] else None),
                'coverage_replications':len(values['coverage']),
            }
        result['scenarios'][label]=summary
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reps',type=int,default=150)
    p.add_argument('--seed',type=int,default=2317)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    record=run(reps=args.reps,seed=args.seed)
    output=json.dumps(record,indent=2)+'\n'
    if args.out:
        args.out.write_text(output)
    print(output)
