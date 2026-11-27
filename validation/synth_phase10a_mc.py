"""Seeded SCM point-estimator stress test; NOT inferential certification.

Run: python -m validation.synth_phase10a_mc --reps 150 --output /tmp/scm.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quasi.synth.scm import SyntheticControl


def simulate(reps: int = 150, seed: int = 20261009) -> dict:
    rng = np.random.default_rng(seed)
    outcomes = {}
    for scenario in ("clean_convex_donors", "posttreatment_donor_contamination"):
        errors, prefit, effective = [], [], []
        failures = 0
        for _ in range(reps):
            n, periods, onset = 6, 16, 10
            t = np.arange(periods)
            common = rng.normal(size=periods).cumsum() * .12
            level = rng.normal(0, .4, size=n)
            slope = rng.normal(.12, .07, size=n)
            donor = (level[:, None] + slope[:, None] * t[None, :] +
                     rng.normal(0, .05, size=(n, periods)) + common[None, :])
            true_counterfactual = .65 * donor[0] + .35 * donor[1]
            treated = true_counterfactual + rng.normal(0, .05, size=periods)
            treated[onset:] += 2.0
            donor_observed = donor.copy()
            if scenario == "posttreatment_donor_contamination":
                donor_observed[0, onset:] += 1.3
            df = pd.DataFrame([(unit, int(period), float(value))
                for unit, arr in [('treated', treated)] + [(f'donor_{i}', x) for i, x in enumerate(donor_observed)]
                for period, value in enumerate(arr)], columns=['unit', 'time', 'y'])
            try:
                result = SyntheticControl(df, 'y', 'unit', 'time', 'treated', onset,
                                          n_placebos=0).fit()
            except Exception:
                failures += 1
                continue
            errors.append(result.estimate - 2.0)
            prefit.append(result.pre_rmspe)
            effective.append(result.model_meta['effective_donors'])
        outcomes[scenario] = {
            "replications_requested": reps, "fits": len(errors), "failures": failures,
            "known_post_average_effect": 2.0,
            "mean_bias": float(np.mean(errors)) if errors else None,
            "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            "mean_pretreatment_rmspe": float(np.mean(prefit)) if prefit else None,
            "mean_effective_donors": float(np.mean(effective)) if effective else None,
        }
    return {"status": "exploratory_not_reference_certified", "seed": seed,
            "scenario_note": "Donor contamination deliberately violates SCM assumptions; clean scenario is a convex donor mixture with observation noise",
            "data_generating_process": "16 periods, 10 pretreatment, 6 donors, effect 2, donor weights 0.65/0.35; stationary observation noise and shared low-frequency trend",
            "scenarios": outcomes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=150)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error('--reps must be positive')
    evidence = simulate(args.reps, args.seed)
    serialized = json.dumps(evidence, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    print(serialized)


if __name__ == '__main__':
    main()
