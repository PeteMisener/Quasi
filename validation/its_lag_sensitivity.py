"""Paired Monte Carlo ITS HAC-lag sensitivity; exploratory, never parity/certification.

Run: PYTHONPATH=src python -m validation.its_lag_sensitivity --reps 240 --output result.json
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
class ITSProfile:
    scenario: str
    lag: int
    replications: int
    successes: int
    failures: int
    coverage: float
    wilson_95: tuple[float, float]
    bias: float
    avg_interval_width: float
    inference_status: str = 'exploratory_unvalidated'


def run(*, reps: int = 240, seed: int = 20261009, lags: tuple[int, ...] = (0, 3, 8, 12)) -> list[ITSProfile]:
    """Use the SAME simulated series at every lag for comparability."""
    if type(reps) is not int or reps < 2 or type(seed) is not int or seed < 0:
        raise ValueError('replications must be >=2 and seed a nonnegative integer')
    if not lags or any(type(lag) is not int or lag < 0 or lag >= 96 for lag in lags):
        raise ValueError('lags must be integers between 0 and 95 for the n=96 DGP')
    import quasi

    seeds = np.random.SeedSequence(seed).spawn(reps)
    scenarios = ('its_ar1', 'its_high_autocorr')
    summary = []
    for scenario in scenarios:
        data = [generate(scenario, int(child.generate_state(1)[0])) for child in seeds]
        for lag in lags:
            successes = 0
            estimates = []
            widths = []
            for exp in data:
                try:
                    res = quasi.InterruptedTimeSeries(exp.data, 'y', 'time', 2048,
                                                      hac_lag=lag, alpha=0.05).fit()
                    if not all(np.isfinite([res.estimate, res.ci_lower, res.ci_upper])):
                        continue
                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError):
                    continue
                successes += int(res.ci_lower <= exp.truth <= res.ci_upper)
                estimates.append(res.estimate - exp.truth)
                widths.append(res.ci_upper - res.ci_lower)
            n = len(estimates)
            summary.append(ITSProfile(scenario, lag, reps, n, reps-n,
                successes / n if n else float('nan'), wilson(successes, n) if n else (float('nan'), float('nan')),
                float(np.mean(estimates)) if n else float('nan'),
                float(np.mean(widths)) if n else float('nan')))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=240)
    ap.add_argument('--seed', type=int, default=20261009)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = [asdict(row) for row in run(reps=args.reps, seed=args.seed)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'seed': args.seed, 'reps': args.reps,
        'notes': 'Paired AR1 DGP simulations; NO inference certification or optimal lag selection.',
        'profiles': rows}, indent=2, allow_nan=False) + '\n')
    for row in rows:
        print(f'{row["scenario"]:<20} lag={row["lag"]:2} coverage={row["coverage"]:.3f} '
              f'interval={row["wilson_95"]!s} failures={row["failures"]}')


if __name__ == '__main__':
    main()
