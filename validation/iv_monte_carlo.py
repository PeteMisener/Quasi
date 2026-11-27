"""Exploratory paired weak/strong IV simulations; never a validation certificate.

Run: PYTHONPATH=src:. python -m validation.iv_monte_carlo --reps 300
"""
from __future__ import annotations
import argparse
import json
from math import sqrt
import numpy as np
import pandas as pd
from scipy import stats

from quasi.iv.tsls import TwoStageLS


def run(*, reps: int = 300, seed: int = 1641, n: int = 200, strengths=(0.04, 0.9)) -> dict:
    rng = np.random.default_rng(seed)
    scenarios = []
    truth = 1.25
    for strength in strengths:
        estimates, wald_rej, ar_rej, fstats, failures = [], [], [], [], 0
        for _ in range(reps):
            # Independent variation in instrument; structural and first-stage
            # shocks correlated, plus heteroskedastic outcome errors.
            z, w, v, noise = rng.normal(size=(4,n))
            x = strength*z + 0.4*w + v
            y = truth*x + 0.5*w + 0.75*v + (0.65+0.30*np.abs(w))*noise
            try:
                fit = TwoStageLS(pd.DataFrame({'y':y,'x':x,'z':z,'w':w}),
                                 'y','x',['z'], covariates=['w']).fit()
            except ValueError:
                failures += 1
                continue
            estimates.append(fit.estimate)
            wald_rej.append(int(abs(fit.estimate-truth)/fit.se > stats.norm.ppf(.975)))
            ar_rej.append(int(fit.ar_test(truth).p_value < 0.05))
            fstats.append(fit.first_stage_robust.statistic)
        m=len(estimates)
        if not m:
            raise RuntimeError('No successful estimates')
        scenarios.append(dict(instrument_strength=strength, replications=reps, successful=m,
             failures=failures, median_estimate=float(np.median(estimates)),
             median_abs_error=float(np.median(np.abs(np.array(estimates)-truth))),
             wald_null_rejection=float(np.mean(wald_rej)), ar_null_rejection=float(np.mean(ar_rej)),
             mean_robust_first_stage_statistic=float(np.mean(fstats)),
             ar_null_rejection_mcse=sqrt(np.mean(ar_rej)*(1-np.mean(ar_rej))/m)))
    return {'analysis':'exploratory_single_endogenous_iv', 'reference_certified':False,
            'true_coefficient':truth, 'seed':seed, 'sample_size':n,
            'nominal_null_rejection':0.05, 'scenarios':scenarios,
            'warnings':['Results depend on this data-generating process.',
                        'The AR test is robust to weak instruments under exogeneity and score-variance assumptions, not to instrument invalidity.',
                        'These simulations do not certify weak-IV-robust confidence sets or cluster small-sample calibration.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1641)
    parser.add_argument('--output', type=str)
    args=parser.parse_args()
    payload=run(reps=args.reps,seed=args.seed)
    data=json.dumps(payload, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(data+'\n')
    else:
        print(data)
