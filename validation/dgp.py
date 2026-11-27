"""Explicit known-truth data-generating processes (NOT external parity datasets)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Experiment:
    data: pd.DataFrame
    truth: float
    scenario: str
    notes: str


def iv(rng: np.random.Generator, *, n: int = 240, truth: float = 1.5,
       strength: float = 0.9, invalid_exclusion: bool = False) -> Experiment:
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    x = strength * z + 0.75 * u + rng.normal(size=n)
    y = 1 + truth * x + 0.8 * u + rng.normal(size=n)
    if invalid_exclusion:
        y = y + 0.6 * z
    return Experiment(pd.DataFrame({"y": y, "x": x, "z": z}), truth, "iv",
                      "Instrument exclusion violated" if invalid_exclusion else "Valid independent instrument")


def its(rng: np.random.Generator, *, n: int = 96, truth: float = 2.0,
        ar: float = 0.3, missing_period: bool = False) -> Experiment:
    t = np.arange(n)
    eps = rng.normal(size=n)
    for i in range(1, n):
        eps[i] += ar * eps[i - 1]
    values = 15 + 0.04 * t + truth * (t >= n // 2) + eps
    df = pd.DataFrame({"time": 2000 + t, "y": values})
    if missing_period:
        df = df.drop(index=n // 3).reset_index(drop=True)
    return Experiment(df, truth, "its", "Segmented level intervention with AR(1) errors")


def did(rng: np.random.Generator, *, units: int = 100, periods: int = 6,
        truth: float = 2.0, anticipation: bool = False) -> Experiment:
    rows = []
    unit_effect = rng.normal(size=units)
    for u in range(units):
        cohort = 3 if u < units // 3 else (4 if u < 2 * units // 3 else None)
        for t in range(periods):
            effect = truth if cohort is not None and t >= cohort else 0.0
            if anticipation and cohort is not None and t == cohort - 1:
                effect = truth / 2
            y = 3 + unit_effect[u] + 0.2 * t + effect + rng.normal()
            rows.append((u, t, int(cohort is not None and t >= cohort), y))
    return Experiment(pd.DataFrame(rows, columns=["unit", "time", "treated", "y"]),
                      truth, "did", "Staggered cohorts; parallel trends" if not anticipation else "Anticipation violation")


def rd(rng: np.random.Generator, *, n: int = 600, truth: float = 1.0,
       mass_points: bool = False, smooth_quadratic: bool = False) -> Experiment:
    x = rng.uniform(-2, 2, n)
    if mass_points:
        x = np.round(x, 1)
    y = 1 + 0.75 * x + (0.6 * x * x if smooth_quadratic else 0) + truth * (x >= 0) + rng.normal(size=n)
    return Experiment(pd.DataFrame({"running": x, "y": y}), truth, "rd",
                      "Smooth cutoff response" if not mass_points else "Discrete running variable mass points")


def synth(rng: np.random.Generator, *, donors: int = 20, periods: int = 30,
          truth: float = 2.0) -> Experiment:
    t = np.arange(periods)
    baseline = rng.normal(size=periods).cumsum() / 10
    rows = []
    for u in range(donors + 1):
        for period in t:
            treated = u == donors and period >= periods // 2
            outcome = 5 + baseline[period] + rng.normal(scale=0.2) + (truth if treated else 0)
            rows.append((u, period, outcome, int(treated)))
    return Experiment(pd.DataFrame(rows, columns=["unit", "time", "y", "treated"]),
                      truth, "synthetic", "One treated unit, common baseline and placebo donor pool")


def overlap(rng: np.random.Generator, *, n: int = 500,
            positivity_violation: bool = False) -> Experiment:
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(12 if positivity_violation else 0.7) * x))
    t = rng.binomial(1, p, size=n)
    y = 1 + x + 1.5 * t + rng.normal(size=n)
    return Experiment(pd.DataFrame({"x": x, "treated": t, "y": y}), 1.5, "overlap",
                      "Near-deterministic treatment assignment" if positivity_violation else "Overlapping treatment probabilities")


SCENARIOS = {
    "iv_strong": (iv, {"truth": 1.5, "strength": 0.9}),
    "iv_null": (iv, {"truth": 0.0, "strength": 0.9}),
    "iv_weak": (iv, {"truth": 1.5, "strength": 0.035}),
    "iv_invalid_exclusion": (iv, {"truth": 1.5, "strength": 0.9, "invalid_exclusion": True}),
    "its_ar1": (its, {"truth": 2.0, "ar": 0.3}),
    "its_null": (its, {"truth": 0.0, "ar": 0.3}),
    "its_high_autocorr": (its, {"truth": 2.0, "ar": 0.85}),
    "did_parallel": (did, {"truth": 2.0}),
    "did_null": (did, {"truth": 0.0}),
    "did_anticipation": (did, {"truth": 2.0, "anticipation": True}),
    "rdd_linear": (rd, {"truth": 1.0}),
    "rdd_quadratic": (rd, {"truth": 1.0, "smooth_quadratic": True}),
    "rdd_mass_points": (rd, {"truth": 1.0, "mass_points": True}),
    "synthetic_shared_trend": (synth, {}),
    "overlap_valid": (overlap, {}),
    "overlap_failure": (overlap, {"positivity_violation": True}),
}


def generate(name: str, seed: int) -> Experiment:
    if name not in SCENARIOS:
        raise ValueError(f"Unsupported DGP scenario: {name}")
    fn, kwargs = SCENARIOS[name]
    return fn(np.random.default_rng(seed), **kwargs)
