"""Gate 1 Algebraic & Statistical Invariance Property Tests across all estimator families."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quasi


def _make_iv(seed=501):
    rng = np.random.default_rng(seed)
    n = 160
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.9 * z + 0.5 * u + rng.normal(size=n)
    y = 2.0 + 1.8 * x + 0.5 * u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _make_did(seed=502):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(40):
        c = 3 if u < 20 else None
        fe = rng.normal()
        for t in range(6):
            d = int(c is not None and t >= c)
            rows.append({"u": u, "t": t, "d": d, "y": 5.0 + fe + 0.2 * t + 2.5 * d + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


def _make_rdd(seed=503):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, 500)
    y = 3.0 + 1.5 * (x >= 0) + 0.4 * x + rng.normal(0, 0.5, len(x))
    return pd.DataFrame({"x": x, "y": y})


def _make_its(seed=504):
    rng = np.random.default_rng(seed)
    t = np.arange(48)
    y = 10.0 + 0.1 * t + 2.5 * (t >= 24) + rng.normal(0, 0.4, len(t))
    return pd.DataFrame({"t": t, "y": y})


def _make_synth(seed=505):
    rng = np.random.default_rng(seed)
    rows = []
    trend = np.linspace(0, 3, 14)
    for u in ("trt", "d1", "d2", "d3", "d4", "d5"):
        off = rng.normal(0, 1.0)
        for t in range(14):
            eff = 2.0 if (u == "trt" and t >= 9) else 0.0
            rows.append({"u": u, "t": t, "y": 5.0 + off + trend[t] + eff + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def _make_obs(seed=506):
    rng = np.random.default_rng(seed)
    n = 260
    x = rng.normal(size=n)
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.6 * x)))
    y = 1.0 + 0.7 * x + 2.0 * d + rng.normal(0, 0.6, n)
    return pd.DataFrame({"y": y, "d": d, "x": x})


CASES = [
    ("iv", _make_iv, lambda: quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",))),
    ("did", _make_did, lambda: quasi.PanelDesign(outcome="y", unit="u", time="t", treatment="d", b_reps=49)),
    ("rdd", _make_rdd, lambda: quasi.RDDDesign(outcome="y", running_var="x", cutoff=0.0, bandwidth=1.0)),
    ("its_hac", _make_its, lambda: quasi.ITSDesign(outcome="y", time="t", intervention_date=24, error_model="hac")),
    ("its_ewc", _make_its, lambda: quasi.ITSDesign(outcome="y", time="t", intervention_date=24, error_model="ewc")),
    ("scm", _make_synth, lambda: quasi.SyntheticDesign(outcome="y", unit="u", time="t", treated_unit="trt", intervention_time=9, estimator="scm", n_placebos=0)),
    ("sdid", _make_synth, lambda: quasi.SyntheticDesign(outcome="y", unit="u", time="t", treated_unit="trt", intervention_time=9, estimator="sdid", n_placebos=0)),
    ("aipw", _make_obs, lambda: quasi.ObservationalDesign(outcome="y", treatment="d", covariates=("x",), estimator="aipw")),
    ("ipw", _make_obs, lambda: quasi.ObservationalDesign(outcome="y", treatment="d", covariates=("x",), estimator="ipw")),
    ("psm", _make_obs, lambda: quasi.ObservationalDesign(outcome="y", treatment="d", covariates=("x",), estimator="psm")),
]


@pytest.mark.parametrize("name,maker,design_fn", CASES, ids=[c[0] for c in CASES])
def test_outcome_shift_invariance_scale_equivariance_and_sign_symmetry(name, maker, design_fn):
    df = maker()
    spec = design_fn()
    base = quasi.estimate(df, design=spec)

    # 1. Shift invariance: Y -> Y + 123.45 leaves estimate & SE identical
    df_shift = df.copy()
    df_shift["y"] = df_shift["y"] + 123.45
    shifted = quasi.estimate(df_shift, design=spec)
    # Note: SCM simplex weights minimize ||Y1 - Y0 w||^2 without intercept; with convex sum(w)=1,
    # shifting ALL units by +c preserves Y1 - Y0 w identically!
    assert shifted.estimate == pytest.approx(base.estimate, rel=1e-5, abs=1e-5)
    if np.isfinite(base.se):
        assert shifted.se == pytest.approx(base.se, rel=1e-5, abs=1e-5)

    # 2. Scale equivariance: Y -> 7.5 * Y scales estimate & SE by 7.5, preserves p-value
    scale = 7.5
    df_scale = df.copy()
    df_scale["y"] = df_scale["y"] * scale
    scaled = quasi.estimate(df_scale, design=spec)
    assert scaled.estimate == pytest.approx(scale * base.estimate, rel=1e-5, abs=1e-5)
    if np.isfinite(base.se):
        assert scaled.se == pytest.approx(scale * base.se, rel=1e-5, abs=1e-5)
        assert scaled.p_value == pytest.approx(base.p_value, rel=1e-5, abs=1e-6)

    # 3. Sign symmetry: Y -> -Y negates estimate and preserves SE
    df_neg = df.copy()
    df_neg["y"] = -df_neg["y"]
    neg = quasi.estimate(df_neg, design=spec)
    assert neg.estimate == pytest.approx(-base.estimate, rel=1e-5, abs=1e-5)
    if np.isfinite(base.se):
        assert neg.se == pytest.approx(base.se, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("name,maker,design_fn", [c for c in CASES if c[0] != "psm"], ids=[c[0] for c in CASES if c[0] != "psm"])
def test_row_order_permutation_invariance(name, maker, design_fn):
    df = maker()
    spec = design_fn()
    base = quasi.estimate(df, design=spec)
    perm = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    reordered = quasi.estimate(perm, design=spec)
    assert reordered.estimate == pytest.approx(base.estimate, rel=1e-6, abs=1e-6)
    if np.isfinite(base.se):
        assert reordered.se == pytest.approx(base.se, rel=1e-6, abs=1e-6)
