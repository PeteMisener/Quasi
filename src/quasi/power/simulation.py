"""Monte Carlo simulation-based power engine for arbitrary Quasi designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

from quasi.designs.specs import DesignSpec
from quasi.errors import InvalidDesignError
from quasi.router import estimate


@dataclass(frozen=True)
class SimulationPowerReport:
    """Empirical power & coverage report from seeded Monte Carlo replications."""

    n_replications: int
    n_succeeded: int
    n_failed: int
    alpha: float
    empirical_power: float
    power_mcse: float
    mean_estimate: float
    mean_se: float
    seed: int

    def summary(self) -> str:
        return (
            f"Simulation Power ({self.n_succeeded}/{self.n_replications} succeeded, seed={self.seed}):\n"
            f"  Empirical power (alpha={self.alpha:g}): {self.empirical_power:.3f} "
            f"(MCSE={self.power_mcse:.4f})\n"
            f"  Mean estimate: {self.mean_estimate:+.4f}; mean SE: {self.mean_se:.4f}"
        )


def simulate_power(
    design: DesignSpec,
    dgp: Callable[[np.random.Generator], pd.DataFrame],
    *,
    n_sims: int = 200,
    seed: int = 42,
) -> SimulationPowerReport:
    """Estimate statistical power by repeatedly generating data and fitting ``design``.

    Failures are counted explicitly and never silently dropped without reporting.
    """
    if type(n_sims) is not int or n_sims < 10:
        raise InvalidDesignError("n_sims must be an integer >= 10")
    if type(seed) is not int or seed < 0:
        raise InvalidDesignError("seed must be a nonnegative integer")
    if not callable(dgp):
        raise InvalidDesignError("dgp must be callable accepting an np.random.Generator")

    children = np.random.SeedSequence(seed).spawn(n_sims)
    rejections = []
    estimates = []
    ses = []
    failures = 0
    alpha = design.alpha

    for child in children:
        rng = np.random.default_rng(child)
        try:
            df = dgp(rng)
            res = estimate(df, design=design)
            if not np.isfinite(res.p_value) or not np.isfinite(res.se):
                failures += 1
                continue
            rejections.append(float(res.p_value < alpha))
            estimates.append(float(res.estimate))
            ses.append(float(res.se))
        except Exception:
            failures += 1

    n_ok = len(rejections)
    if n_ok == 0:
        raise InvalidDesignError("All Monte Carlo power replications failed to produce finite inference")
    pwr = float(np.mean(rejections))
    mcse = float(np.sqrt(pwr * (1.0 - pwr) / n_ok))
    return SimulationPowerReport(
        n_replications=n_sims,
        n_succeeded=n_ok,
        n_failed=failures,
        alpha=alpha,
        empirical_power=pwr,
        power_mcse=mcse,
        mean_estimate=float(np.mean(estimates)),
        mean_se=float(np.mean(ses)),
        seed=seed,
    )
