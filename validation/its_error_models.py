"""Paired exploratory simulation for ITS error models; NOT parity/certification.

Usage: PYTHONPATH=src python -m validation.its_error_models --reps 400 \\
  --output validation/evidence/its_error_models.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from validation.dgp import generate
from validation.monte_carlo import wilson


@dataclass(frozen=True)
class Profile:
    scenario: str
    error_model: str
    replications: int
    succeeded: int
    failed: int
    true_effect: float
    coverage: float
    coverage_wilson_95: tuple[float, float]
    bias: float
    rmse: float
    avg_interval_width: float
    median_fitted_rho: float | None
    inference_status: str = 'exploratory_unvalidated'


def run(*, reps: int = 400, seed: int = 20261009) -> list[Profile]:
    if type(reps) is not int or reps < 2 or type(seed) is not int or seed < 0:
        raise ValueError('reps>=2 and nonnegative seed required')
    import quasi
    output = []
    for scenario in ('its_ar1', 'its_high_autocorr', 'its_null'):
        children = np.random.SeedSequence(seed).spawn(reps)
        data = [generate(scenario, int(child.generate_state(1)[0])) for child in children]
        for mode in ('hac', 'ar1'):
            biases, widths, hits, rhos = [], [], 0, []
            for experiment in data:
                try:
                    model = quasi.InterruptedTimeSeries(experiment.data, 'y', 'time', 2048,
                        hac_lag=3 if mode=='hac' else None, error_model=mode, alpha=.05).fit()
                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError):
                    continue
                if not np.isfinite([model.estimate, model.ci_lower, model.ci_upper]).all():
                    continue
                biases.append(model.estimate-experiment.truth)
                widths.append(model.ci_upper-model.ci_lower)
                hits += int(model.ci_lower<=experiment.truth<=model.ci_upper)
                if mode=='ar1':
                    rhos.append(model.inference_details['estimated_rho'])
            n = len(biases)
            output.append(Profile(scenario, mode, reps, n, reps-n, data[0].truth,
                hits/n if n else float('nan'), wilson(hits,n) if n else (float('nan'),float('nan')),
                float(np.mean(biases)) if n else float('nan'),
                float(np.sqrt(np.mean(np.square(biases)))) if n else float('nan'),
                float(np.mean(widths)) if n else float('nan'),
                float(np.median(rhos)) if rhos else None))
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=400)
    ap.add_argument('--seed',type=int,default=20261009)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    profiles=[asdict(profile) for profile in run(reps=args.reps,seed=args.seed)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'replications':args.reps,'seed':args.seed,
          'reference_certified':False,'notes':'Paired DGP, AR1 conditional covariance treats estimated rho as fixed; exploratory only',
          'profiles':profiles}, indent=2, allow_nan=False)+'\n')
    for item in profiles:
        print(f"{item['scenario']:<21} {item['error_model']:<4} coverage={item['coverage']:.3f} "
              f"CI={item['coverage_wilson_95']} failures={item['failed']}")

if __name__=='__main__':
    main()
