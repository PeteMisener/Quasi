"""Exploratory reproducible DR-ATT calibration; NOT certification.

Run: PYTHONPATH=src python -m validation.observational.monte_carlo_phase11 --reps 200
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from quasi.matching.psm import AIPWATT


def generate(rng: np.random.Generator, n: int, scenario: str) -> pd.DataFrame:
    x = rng.normal(size=n)
    if scenario == "propensity_correct_outcome_misspecified":
        e = expit(0.2 + 0.8*x)
        untreated = 1 + 0.8*x + 0.8*(x*x - 1)
    elif scenario == "outcome_correct_propensity_misspecified":
        e = expit(0.1 + 0.8*x + 0.7*(x*x-1))
        untreated = 1 + 0.8*x
    elif scenario == "both_correct":
        e = expit(0.2 + 0.8*x)
        untreated = 1 + 0.8*x
    elif scenario == "both_misspecified":
        e = expit(0.1 + 0.8*x + 0.7*(x*x-1))
        untreated = 1 + 0.8*x + 0.8*(x*x-1)
    else:
        raise ValueError(scenario)
    d = rng.binomial(1,e,size=n)
    y = untreated + 2*d + rng.normal(0, 0.75,size=n)
    return pd.DataFrame(dict(x=x,d=d,y=y))


def main(reps=200, n=800, seed=31415, output=None):
    scenarios = ("both_correct", "propensity_correct_outcome_misspecified",
                 "outcome_correct_propensity_misspecified", "both_misspecified")
    rng = np.random.default_rng(seed)
    report = {"experiment": "Phase11A_exploratory_AIPW_ATT", "method": "stacked_HC0_no_crossfitting",
              "target_ATT":2.0,"repetitions_per_scenario":reps,"n_per_sample":n,"seed":seed,
              "certified":False,"results":{}}
    for scenario in scenarios:
        estimates=[]; intervals=[]; ses=[]; failures={}
        for i in range(reps):
            data=generate(rng,n,scenario)
            try:
                r=AIPWATT(data, outcome='y', treatment='d', covariates=['x']).fit()
            except Exception as err:
                failures[type(err).__name__]=failures.get(type(err).__name__,0)+1
                continue
            estimates.append(r.estimate)
            ses.append(r.se)
            intervals.append(r.ci_lower <= 2.0 <= r.ci_upper)
        report["results"][scenario] = {
            "success": len(estimates), "failures_by_type": failures,
            "mean_bias":float(np.mean(estimates)-2) if estimates else None,
            "rmse":float(np.sqrt(np.mean((np.array(estimates)-2)**2))) if estimates else None,
            "mean_SE":float(np.mean(ses)) if ses else None,
            "CI_coverage":float(np.mean(intervals)) if intervals else None,
            "coverage_monte_carlo_SE":float(np.sqrt(np.mean(intervals)*(1-np.mean(intervals))/len(intervals))) if intervals else None,
        }
    payload=json.dumps(report,indent=2,allow_nan=False)
    if output: Path(output).write_text(payload+'\n')
    print(payload)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=200)
    parser.add_argument('--n',type=int,default=800)
    parser.add_argument('--seed',type=int,default=31415)
    parser.add_argument('--output')
    args=parser.parse_args()
    main(**vars(args))
