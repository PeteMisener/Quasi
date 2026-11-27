"""Exploratory cross-fitting simulation; not statistical certification.

Run: PYTHONPATH=src python -m validation.observational_phase11c_crossfit_mc --reps 160 --output output.json
All invalid or failed fits count towards failure rates; no conditional-only reporting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

from quasi.matching.psm import AIPWATT


def simulate(*, reps: int = 160, n: int = 460, seed: int = 2110) -> dict:
    rng = np.random.default_rng(seed)
    cases = ("linear_outcome", "nonlinear_outcome_correct_propensity", "limited_overlap")
    output = {"configuration": {"seed": seed, "n": n, "reps_per_scenario": reps,
                  "true_att": 2.0, "fold_count": 3, "crossfit_seed": 91,
                  "confidence_level": 0.95}, "scenarios": {}}
    for scenario in cases:
        pairs = {"unsplit": [], "crossfit": []}
        failures = {"unsplit": 0, "crossfit": 0}
        for _ in range(reps):
            x = rng.normal(size=n)
            slope = 2.6 if scenario == "limited_overlap" else 0.7
            propensity = expit(-0.25 + slope*x)
            d = rng.binomial(1, propensity)
            nonlinear = 0.75*x*x if scenario == "nonlinear_outcome_correct_propensity" else 0.0
            y = 0.5 + 0.7*x + nonlinear + 2.0*d + rng.normal(scale=1.0, size=n)
            frame = pd.DataFrame({"y":y, "d":d, "x":x})
            for label, n_splits in (("unsplit", 1), ("crossfit", 3)):
                try:
                    result = AIPWATT(frame, "y", "d", ["x"], n_splits=n_splits, fold_seed=91).fit()
                    pairs[label].append((result.estimate, result.se, result.ci_lower <= 2.0 <= result.ci_upper))
                except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                    failures[label] += 1
        output["scenarios"][scenario] = {}
        for label in pairs:
            vals = np.asarray(pairs[label], dtype=float).reshape(-1,3)
            m = len(vals)
            output["scenarios"][scenario][label] = {
                "attempted": reps, "successes": m, "failures": failures[label],
                "failure_rate": failures[label]/reps,
                "mean_bias_if_accepted": float(np.mean(vals[:,0]-2.0)) if m else None,
                "rmse_if_accepted": float(np.sqrt(np.mean((vals[:,0]-2.0)**2))) if m else None,
                "nominal_95pct_coverage_if_accepted": float(np.mean(vals[:,2])) if m else None,
                "monte_carlo_coverage_se_if_accepted": float(np.sqrt(np.mean(vals[:,2])*(1-np.mean(vals[:,2]))/m)) if m else None,
                "mean_reported_se_if_accepted": float(np.mean(vals[:,1])) if m else None,
            }
    output["interpretation"] = ("Simulated DGP comparison only. Coverage conditional on successful estimation "
       "is not an unconditional performance guarantee. Orthogonal-score variance requires nuisance-rate conditions. "
       "Cross-fitting does not correct misspecified models or establish unconfoundedness.")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=160)
    parser.add_argument("--n", type=int, default=460)
    parser.add_argument("--seed", type=int, default=2110)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.n < 25:
        parser.error("reps >= 1 and n >= 25 required")
    result = simulate(reps=args.reps, n=args.n, seed=args.seed)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result['scenarios'], indent=2))


if __name__ == "__main__":
    main()
