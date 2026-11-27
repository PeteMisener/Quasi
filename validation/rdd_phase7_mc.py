"""Exploratory Monte Carlo for Phase 7. Never a certification tool.

Run from repo root: PYTHONPATH=src:. python validation/rdd_phase7_mc.py --reps 300
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from quasi.rdd.local_inference import rd_inference


def run(reps=300, n=700, seed=1009):
    rng = np.random.default_rng(seed)
    scenarios = {"linear": 0.0, "curved": 2.6}
    output = {'simulation_type':'synthetic exploratory', 'reference_certified':False,
              'seed':seed, 'reps':reps, 'n':n, 'nominal_level':0.95, 'scenarios':{}}
    for scenario, curvature in scenarios.items():
        vals = {'conventional':[], 'bias_corrected':[]}
        for _ in range(reps):
            x = rng.uniform(-1.6, 1.6, n)
            y = (1.0 + 1.5*(x >= 0) + .6*x + curvature*(x >= 0)*x*x +
                 rng.normal(0, .8, n))
            for method, b in [('conventional',None),('bias_corrected',1.35)]:
                fit = rd_inference(x,y,1.15,'triangular',1,bias_bandwidth=b)
                vals[method].append((fit['estimate'],fit['se']))
        output['scenarios'][scenario] = {}
        for method, array in vals.items():
            arr = np.asarray(array)
            bias = float(np.mean(arr[:,0]-1.5))
            cover = np.abs((arr[:,0]-1.5)/arr[:,1]) <= norm.ppf(.975)
            coverage = float(cover.mean())
            output['scenarios'][scenario][method] = {
                'bias':bias,'rmse':float(np.sqrt(np.mean((arr[:,0]-1.5)**2))),
                'coverage_95':coverage,'coverage_mc_se':float(np.sqrt(coverage*(1-coverage)/reps)),
                'mean_se':float(np.mean(arr[:,1])),
            }
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=300)
    parser.add_argument('--output',type=Path,default=Path('/mnt/data/Quasi_Iteration_9_RDD_Monte_Carlo.json'))
    args = parser.parse_args()
    result = run(reps=args.reps)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
