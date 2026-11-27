"""Seeded simulation experiment runner; *not* a reference-parity validator.

Coverage and type-I error are empirical diagnostics, not certification. All
failed/nonfinite replicates are counted, and no thresholds are tuned to pass.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from validation.dgp import generate


@dataclass(frozen=True)
class SimulationSummary:
    scenario: str
    seed: int
    requested_reps: int
    completed_reps: int
    failed_reps: int
    truth: float
    nominal_coverage: float
    bias: float | None
    bias_mcse: float | None
    rmse: float | None
    coverage: float | None
    coverage_mcse: float | None
    coverage_wilson_95: tuple[float, float] | None
    rejection_rate: float | None
    rejection_mcse: float | None
    average_interval_width: float | None
    failures: list[str]
    status: str


def wilson(k: int, n: int, *, confidence_level: float = 0.95) -> tuple[float, float]:
    if not (0 <= k <= n) or n < 1:
        raise ValueError("Wilson interval requires 0 <= k <= n, with n >= 1")
    z = float(norm.ppf((1 + confidence_level) / 2))
    p = k / n
    denominator = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / denominator
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (mid - rad, mid + rad)


def _fit(experiment, name: str, alpha: float):
    import quasi

    if name.startswith("iv_"):
        return quasi.TwoStageLS(experiment.data, "y", "x", ["z"], alpha=alpha).fit()
    if name.startswith("its_"):
        return quasi.InterruptedTimeSeries(experiment.data, "y", "time", 2048, hac_lag=3, alpha=alpha).fit()
    if name.startswith("did_"):
        return quasi.StaggeredDiD(experiment.data, "y", "unit", "time",
                                   treatment="treated", b_reps=199, seed=711,
                                   alpha=alpha).fit()
    if name.startswith("rdd_"):
        return quasi.Discontinuity(experiment.data, "y", "running", 0.0, bandwidth=1.0,
                                    order=1, kernel="triangular", alpha=alpha).fit()
    raise ValueError(f"Scenario {name} is fixture-only (estimator automation not yet validated)")


def run_simulation(name: str, *, reps: int = 100, seed: int = 20261009,
                   alpha: float = 0.05) -> SimulationSummary:
    if reps < 2 or reps > 100000 or not (0 < alpha < 1):
        raise ValueError("Require 2 <= reps <= 100000 and 0 < alpha < 1")
    if not name.startswith(("iv_", "its_", "rdd_", "did_")):
        raise ValueError("Only IV/ITS/RDD/DiD automatic fits are currently supported; other DGPs are fixture-only")
    # Distinct deterministic substreams make runs reproducible and order-independent.
    generator = np.random.SeedSequence(seed)
    rep_seeds = generator.spawn(reps)
    values, coverages, rejections, widths, failures = [], [], [], [], []
    truth = None
    for idx, child in enumerate(rep_seeds):
        exp = generate(name, int(child.generate_state(1)[0]))
        truth = exp.truth
        try:
            result = _fit(exp, name, alpha)
            estimate, lo, hi, pval = (float(result.estimate), float(result.ci_lower),
                                      float(result.ci_upper), float(result.p_value))
            if not (all(map(math.isfinite, [estimate, lo, hi, pval])) and lo <= hi and 0 <= pval <= 1):
                raise ValueError("Nonfinite or malformed estimate/inference")
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            failures.append(f"rep={idx}: {type(exc).__name__}: {exc}")
            continue
        values.append(estimate)
        coverages.append(int(lo <= exp.truth <= hi))
        rejections.append(int(pval < alpha))
        widths.append(hi - lo)
    n = len(values)
    if n:
        errs = np.asarray(values) - truth
        covered = sum(coverages)
        rejected = sum(rejections)
        coverage = covered / n
        rejection_rate = rejected / n
        values_arr = np.asarray(values)
        return SimulationSummary(
            name, seed, reps, n, reps - n, float(truth), 1 - alpha,
            float(np.mean(errs)), float(np.std(errs, ddof=1) / np.sqrt(n)) if n > 1 else None,
            float(np.sqrt(np.mean(errs ** 2))), coverage,
            float(np.sqrt(coverage * (1 - coverage) / n)), wilson(covered, n),
            rejection_rate, float(np.sqrt(rejection_rate * (1 - rejection_rate) / n)),
            float(np.mean(widths)), failures[:20], "experimental_unreviewed",
        )
    return SimulationSummary(name, seed, reps, 0, reps, float(truth), 1 - alpha,
                             None, None, None, None, None, None, None, None, None,
                             failures[:20], "all_replicates_failed")


def write_report(results: list[SimulationSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2, allow_nan=False) + "\n")
