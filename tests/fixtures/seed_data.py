"""
Deterministic fixture data and embedded golden reference benchmarks for quasi.
Enables offline synthetic smoke tests; does NOT establish reference parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent


# ------------------------------------------------------------------------------
# 1. Unverified embedded reference targets (NOT used by synthetic fallback)
# ------------------------------------------------------------------------------

MPDTA_EMPIRICAL_GOLDEN: Dict[str, Any] = {
    "package": "did",
    "dataset": "mpdta",
    "overall_att": -0.03988,
    "overall_se": 0.01257,
    "p_value": 0.00151,
}

R_SENATE_EMPIRICAL_GOLDEN: Dict[str, Any] = {
    "package": "rdrobust",
    "dataset": "rdrobust_RDsenate",
    "cutoff": 0.0,
    "h_optimal": 17.754,
    "estimate": 7.414,
    "se_conventional": 1.452,
    "se_robust": 1.718,
    "p_value_robust": 0.00002,
}

PROP99_EMPIRICAL_GOLDEN: Dict[str, Any] = {
    "package": "synthdid",
    "dataset": "california_prop99",
    "treated_unit": "California",
    "intervention_time": 1989,
    "estimate": -27.349,
    "se_jackknife": 9.421,
    "p_value": 0.0037,
}


# ------------------------------------------------------------------------------
# 2. Deterministic Seed Benchmark Targets (Exact values for seed=42)
# ------------------------------------------------------------------------------

MPDTA_SYNTHETIC_GOLDEN: Dict[str, Any] = {
    "package": "did",
    "dataset": "mpdta_synthetic",
    "overall_att": -0.0449,
    "overall_se": 0.0126,
    "p_value": 0.00035,
}

R_SENATE_SYNTHETIC_GOLDEN: Dict[str, Any] = {
    "package": "rdrobust",
    "dataset": "rdrobust_RDsenate_synthetic",
    "cutoff": 0.0,
    "h_optimal": 8.557,
    "estimate": 9.259,
    "se_conventional": 1.854,
    "p_value": 0.00001,
}

PROP99_SYNTHETIC_GOLDEN: Dict[str, Any] = {
    "package": "synthdid",
    "dataset": "california_prop99_synthetic",
    "treated_unit": "California",
    "intervention_time": 1989,
    "estimate": -25.343,
    "se_jackknife": 0.699,
    "p_value": 0.00001,
}


# ------------------------------------------------------------------------------
# 3. Deterministic Dataset Generators (seed=42)
# ------------------------------------------------------------------------------

def get_mpdta_dataset(n_counties: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates deterministic panel data matching Callaway & Sant'Anna setup."""
    rng = np.random.default_rng(seed)
    years = [2003, 2004, 2005, 2006, 2007]
    cohorts = [2004, 2006, 2007, 0]

    county_ids = np.arange(1, n_counties + 1)
    g_assignments = rng.choice(cohorts, size=n_counties, p=[0.25, 0.20, 0.20, 0.35])
    base_employment = rng.normal(loc=5.5, scale=0.4, size=n_counties)

    rows = []
    for i, cid in enumerate(county_ids):
        g = g_assignments[i]
        b_emp = base_employment[i]
        for t in years:
            time_trend = 0.02 * (t - 2003)
            treatment_active = 1 if (g > 0 and t >= g) else 0

            dynamic_effect = 0.0
            if treatment_active:
                rel_t = t - g
                dynamic_effect = -0.0211 - 0.016 * rel_t

            noise = rng.normal(loc=0.0, scale=0.128)
            log_emp = b_emp + time_trend + dynamic_effect + noise

            rows.append({
                "countyreal": cid,
                "year": t,
                "first_treat": g,
                "treat": treatment_active,
                "lemp": log_emp,
            })

    return pd.DataFrame(rows)


def get_senate_dataset(n_obs: int = 1297, seed: int = 42) -> pd.DataFrame:
    """Generates deterministic US Senate election discontinuity data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=1.2, scale=18.5, size=n_obs)
    D = (X >= 0.0).astype(float)
    x_c = X
    noise = rng.normal(0.0, 9.2, size=n_obs)
    Y = 46.5 + 0.42 * x_c + (7.414 * D) + (0.05 * x_c * D) - 0.0005 * (x_c**2) + noise

    return pd.DataFrame({
        "vote_margin_t": X,
        "dem_incumbent": D,
        "vote_share_t1": Y,
    })


def get_prop99_dataset(n_donors: int = 38, seed: int = 42) -> pd.DataFrame:
    """Generates deterministic Proposition 99 state panel data."""
    rng = np.random.default_rng(seed)
    years = list(range(1970, 2001))
    states = ["California"] + [f"State_{i:02d}" for i in range(1, n_donors + 1)]

    base_consumption = {s: rng.normal(120.0, 15.0) for s in states}
    base_consumption["California"] = 118.5
    national_trend = np.linspace(0.0, -40.0, len(years))

    state_trends = {s: rng.normal(0.0, 0.45, size=len(years)).cumsum() for s in states}
    state_trends["California"] = np.zeros(len(years))

    rows = []
    for s in states:
        b = base_consumption[s]
        state_factor = rng.normal(1.0, 0.08)
        s_trend = state_trends[s]
        for t_idx, yr in enumerate(years):
            noise = rng.normal(0, 3.5)
            policy_drop = 0.0
            if s == "California" and yr >= 1989:
                policy_drop = -27.349

            sales = b + national_trend[t_idx] * state_factor + s_trend[t_idx] + policy_drop + noise
            rows.append({
                "state": s,
                "year": yr,
                "prop99_active": 1 if (s == "California" and yr >= 1989) else 0,
                "packs_per_capita": sales,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 4. Legacy fall-back loaders (NOT valid parity fixtures)
# ------------------------------------------------------------------------------

def get_mpdta_benchmark() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    csv_path = FIXTURES_DIR / "mpdta.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "first.treat" in df.columns and "first_treat" not in df.columns:
            df["first_treat"] = df["first.treat"]
        return df, MPDTA_EMPIRICAL_GOLDEN
    return get_mpdta_dataset(seed=42), MPDTA_SYNTHETIC_GOLDEN


def get_senate_benchmark() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    csv_path = FIXTURES_DIR / "rdrobust_RDsenate.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df, R_SENATE_EMPIRICAL_GOLDEN
    return get_senate_dataset(seed=42), R_SENATE_SYNTHETIC_GOLDEN


def get_prop99_benchmark() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    csv_path = FIXTURES_DIR / "california_prop99.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df, PROP99_EMPIRICAL_GOLDEN
    return get_prop99_dataset(seed=42), PROP99_SYNTHETIC_GOLDEN


def ensure_benchmark_files_exist() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = {
        "mpdta_benchmark.json": MPDTA_SYNTHETIC_GOLDEN,
        "r_senate_benchmark.json": R_SENATE_SYNTHETIC_GOLDEN,
        "prop99_benchmark.json": PROP99_SYNTHETIC_GOLDEN,
    }
    for filename, content in benchmarks.items():
        file_path = FIXTURES_DIR / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)