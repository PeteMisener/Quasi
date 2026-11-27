"""Regression tests for published rdrobust fuzzy-RBC linearization.

These use independently fitted Statsmodels WLS side regressions for HC0
residuals; not a substitute for executing the external rdrobust package.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import quasi
from quasi.rdd.local_inference import boundary_weights, rd_inference


def _curved_fuzzy(seed=146, n=2600):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.8, 1.8, n)
    # Curved, heterogeneous treatment propensity on both sides; nonzero
    # jumps in the uncorrected and bias-corrected first stages.
    p = np.clip(.22 + .49 * (x >= 0) + .09 * x + .16 * x*x * (x >= 0), .05, .94)
    t = (rng.uniform(size=n) < p).astype(float)
    y = 2. + 1.8 * t + .7*x + .9*x*x*(x >= 0) + rng.normal(0, .55, n)
    return pd.DataFrame({'x': x, 'd': t, 'y': y})


def _linearized_formula(df, h, b):
    x, y, t = (df[k].to_numpy() for k in ('x', 'y', 'd'))
    R = boundary_weights(x, h, b, 1, 'triangular', 'right')
    L = boundary_weights(x, h, b, 1, 'triangular', 'left')
    w_p = R.conventional - L.conventional
    w_bc = R.corrected - L.corrected
    y_p, t_p = float(w_p @ y), float(w_p @ t)
    y_bc, t_bc = float(w_bc @ y), float(w_bc @ t)
    tau_p = y_p/t_p
    # First-order, not the ratio y_bc/t_bc.
    tau_bc = tau_p - (y_p-y_bc)/t_p + y_p*(t_p-t_bc)/t_p**2
    return tau_p, tau_bc, y_bc/t_bc, w_bc, t_p


def test_fuzzy_bias_correction_is_first_order_not_ratio():
    df = _curved_fuzzy()
    h, b = 1.0, 1.4
    conventional, corrected, naive_ratio, _, _ = _linearized_formula(df, h, b)
    output = rd_inference(df.x.to_numpy(), df.y.to_numpy(), h, 'triangular', 1,
                          treatment=df.d.to_numpy(), bias_bandwidth=b)
    assert output['estimate'] == pytest.approx(corrected, abs=1e-12)
    assert output['conventional_estimate'] == pytest.approx(conventional, abs=1e-12)
    assert abs(corrected - naive_ratio) > 1e-5
    assert output['fuzzy_bias_correction'] == 'conventional_wald_first_order'


def test_fuzzy_corrected_se_uses_conventional_wald_gradient():
    df = _curved_fuzzy()
    x, y, t = (df[k].to_numpy() for k in ('x', 'y', 'd'))
    h, b = .95, 1.35
    tau, _, _, weights, stage = _linearized_formula(df, h, b)
    output = rd_inference(x, y, h, 'triangular', 1,
                          treatment=t, bias_bandwidth=b)
    residual = np.zeros(len(x))
    for side in (x < 0, x >= 0):
        mask = side & (np.abs(x) < b)
        Z = np.column_stack([np.ones(mask.sum()), x[mask]/b, (x[mask]/b)**2])
        fit = sm.WLS((y-tau*t)[mask], Z, weights=1-np.abs(x[mask])/b).fit()
        residual[mask] = fit.resid
    manual_se = np.sqrt(np.sum((weights * residual / stage)**2))
    assert output['se'] == pytest.approx(manual_se, rel=1e-10)


def test_public_fuzzy_metadata_records_first_order_correction():
    df = _curved_fuzzy(n=1400)
    fit = quasi.FuzzyRDD(df, 'y', 'x', 0, treatment='d', bandwidth=1.2,
                         inference='bias_corrected', bias_bandwidth=1.45).fit()
    assert fit.inference_details['fuzzy_bias_correction'] == 'conventional_wald_first_order'
    assert fit.inference_details['reference_certified'] is False
    assert np.isfinite(fit.se) and fit.se > 0
