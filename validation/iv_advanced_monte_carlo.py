"""Experimental IV phase 8B Monte Carlo, *not* method certification.

Seeded simulations check AR confidence-set coverage against weak/strong and
invalid-instrument designs. Exact one-instrument inversion may be unbounded.
Results are exploratory; the finite-sample validity of HC/CR AR calibration is
not established by numerical agreement alone.

Run: PYTHONPATH=src:. python -m validation.iv_advanced_monte_carlo --reps 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quasi.iv.tsls import TwoStageLS


def run(*, reps: int = 400, n: int = 200, seed: int = 1515) -> dict:
    if type(reps) is not int or reps < 1 or type(n) is not int or n < 25:
        raise ValueError('Need at least one replication and 25 observations')
    rng = np.random.default_rng(seed)
    true_effect = 1.25
    scenarios = {}
    for name, strength, invalid in [
        ('valid_weak', 0.025, False),
        ('valid_strong', 0.9, False),
        ('invalid_exclusion', 0.9, True),
    ]:
        covered_ar, covered_wald, unbounded, rejected_ar, rejected_wald = [], [], [], [], []
        estimates = []
        for _ in range(reps):
            z, w, v, err = rng.normal(size=(4, n))
            x = strength*z + 0.4*w + v
            y = true_effect*x + 0.5*w + 0.7*v + err*(0.5+0.4*np.abs(w))
            if invalid:
                y = y + 0.45*z
            df = pd.DataFrame({'y':y,'x':x,'z':z,'w':w})
            result = TwoStageLS(df,'y','x',['z'],covariates=['w']).fit()
            cs = result.ar_confidence_set()
            covered_ar.append(cs.contains(true_effect))
            covered_wald.append(result.ci_lower <= true_effect <= result.ci_upper)
            unbounded.append(cs.is_unbounded)
            rejected_ar.append(result.ar_test(true_effect).p_value < .05)
            rejected_wald.append(not covered_wald[-1])
            estimates.append(result.estimate)
        scenarios[name] = {
            'strength_coefficient': strength, 'invalid_exclusion': invalid,
            'ar_95_coverage': float(np.mean(covered_ar)),
            'wald_95_coverage': float(np.mean(covered_wald)),
            'ar_unbounded_fraction': float(np.mean(unbounded)),
            'ar_true_effect_rejection_rate': float(np.mean(rejected_ar)),
            'wald_true_effect_rejection_rate': float(np.mean(rejected_wald)),
            'median_coefficient': float(np.median(estimates)),
            'monte_carlo_se_at_95pct': float(np.sqrt(.95*.05/reps)),
            'failed_replications': 0,
        }
    return {
        'experiment': 'experimental_one_instrument_AR_inversion',
        'reps_per_scenario':reps,'observations':n,'seed':seed,
        'true_coefficient':true_effect,
        'nominal_coverage':.95,
        'reference_certified':False,
        'methods': 'HC0 asymptotic chi-square AR inversion; HC0 normal Wald',
        'scenarios':scenarios,
        'limits': [
            'Synthetic data only; no independent package numerical parity',
            'AR inversion assumes valid instruments and large-sample HC0 calibration',
            'Only one excluded instrument, one endogenous variable, no clustering',
            'Invalid exclusion scenario deliberately breaks a causal assumption',
        ],
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=400)
    parser.add_argument('--n',type=int,default=200)
    parser.add_argument('--seed',type=int,default=1515)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=run(reps=args.reps,n=args.n,seed=args.seed)
    report=json.dumps(result,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(report+'\n',encoding='utf-8')
    print(report)

if __name__=='__main__':
    main()
