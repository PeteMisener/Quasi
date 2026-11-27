"""Regression tests for critical and high findings identified in the full code review."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quasi
from quasi.did.sensitivity import honest_did_sensitivity
from quasi.errors import InvalidDesignError, UnsupportedInferenceError
from quasi.inference.covariance import cr2_covariance, prewhitened_hac_covariance
from quasi.inference.linear import least_squares
from quasi.inference.optimization import solve_simplex_regression
from quasi.iv.inference import effective_f_statistic, tf_critical_value
from quasi.matching.psm import AIPWATT, PropensityScoreMatching
from quasi.prep.audit import audit_panel
from quasi.reporting.table import ModelComparisonTable, TableFormattingOptions
from quasi.synth.ascm import AugmentedSCM
from quasi.synth.scm import SyntheticControl
from quasi.synth.sdid import SyntheticDiD


def test_ascm_weights_sum_to_one_and_are_shift_invariant() -> None:
    rng = np.random.default_rng(101)
    T_pre, T_post, N0 = 14, 4, 8
    T = T_pre + T_post
    f = rng.normal(0, 1, (T, 2))
    lam = rng.normal(0, 1, (N0, 2))
    Y0 = lam @ f.T + rng.normal(0, 0.3, (N0, T))
    y1 = 0.5 * Y0[0] + 0.5 * Y0[1] + 1.2 + rng.normal(0, 0.2, T)
    y1[T_pre:] += 3.0

    def _build(shift: float) -> pd.DataFrame:
        rows = []
        for t in range(T):
            rows.append(("trt", t, float(y1[t] + shift)))
            for j in range(N0):
                rows.append((f"d{j}", t, float(Y0[j, t] + shift)))
        return pd.DataFrame(rows, columns=["unit", "time", "y"])

    r0 = AugmentedSCM(_build(0.0), "y", "unit", "time", "trt", T_pre).fit()
    r_shift = AugmentedSCM(_build(1000.0), "y", "unit", "time", "trt", T_pre).fit()

    assert sum(r0.donor_weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert r0.estimate == pytest.approx(r_shift.estimate, abs=1e-6)


def test_conformal_full_series_preserves_null_size_and_matches_sdid_estimate() -> None:
    rng = np.random.default_rng(202)
    T_pre, T_post, N0 = 20, 2, 6
    T = T_pre + T_post
    Y0 = rng.normal(10.0, 2.0, (N0, T))
    y1 = 0.6 * Y0[0] + 0.4 * Y0[1] + rng.normal(0, 0.5, T)
    y1[T_pre:] += 5.0
    rows = [("trt", t, float(y1[t])) for t in range(T)] + [
        (f"d{j}", t, float(Y0[j, t])) for j in range(N0) for t in range(T)
    ]
    df = pd.DataFrame(rows, columns=["unit", "time", "y"])
    sdid = SyntheticDiD(df, "y", "unit", "time", "trt", T_pre, n_placebos=4, seed=1, sparsify="default").fit()
    ci = sdid.conformal_interval(alpha=0.10)
    assert ci.estimate == pytest.approx(sdid.estimate, abs=1e-12)
    assert np.isfinite(ci.ci_lower) and np.isfinite(ci.ci_upper)
    assert "sparsification was applied" in sdid.methods_text()
    # Placebo gaps must be intercept-centered around 0 in pre-period
    for gap in sdid._placebo_gaps:
        assert abs(float(np.mean(gap[:T_pre]))) < 2.0


def test_effective_f_mop_critical_values_do_not_blow_up_for_large_k() -> None:
    rng = np.random.default_rng(303)
    n, k = 400, 10
    Z = rng.normal(0, 1, (n, k))
    W = np.ones((n, 1))
    x_end = Z @ np.full(k, 0.3) + rng.normal(0, 1, n)
    eff = effective_f_statistic(x_end, Z, W, kind="HC0")
    # MOP (2013) worst-case 10% bias critical value at K=10 is ~14-18 (previously blew up to ~101!)
    assert 10.0 < eff.critical_values_tsls_bias["10%"] < 25.0
    assert 18.0 < eff.critical_values_tsls_bias["5%"] < 40.0


def test_tf_critical_value_diverges_smoothly_near_pole() -> None:
    c_4 = tf_critical_value(4.0, alpha=0.05)
    c_39 = tf_critical_value(3.90, alpha=0.05)
    c_385 = tf_critical_value(3.85, alpha=0.05)
    assert c_385 > c_39 > c_4
    assert c_385 > 40.0
    assert tf_critical_value(3.84, alpha=0.05) == float("inf")


def test_solve_simplex_regression_is_scale_invariant_for_small_rates() -> None:
    rng = np.random.default_rng(404)
    A = rng.normal(0, 1, (20, 5))
    w_true = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    b = A @ w_true + rng.normal(0, 0.05, 20)
    s1 = solve_simplex_regression(A, b, ridge=0.0, intercept=False)
    s_small = solve_simplex_regression(A * 1e-5, b * 1e-5, ridge=0.0, intercept=False)
    np.testing.assert_allclose(s1.weights, s_small.weights, atol=1e-4)


def test_prewhitened_hac_qs_lag_zero_and_scale_invariance() -> None:
    rng = np.random.default_rng(505)
    T = 80
    X = np.column_stack([np.ones(T), np.arange(T, dtype=float), rng.normal(0, 1, T)])
    u = rng.normal(0, 1, T)
    cov0 = prewhitened_hac_covariance(X, u, kernel="qs", max_lag=0)
    assert np.all(np.isfinite(cov0)) and np.all(np.diag(cov0) > 0)

    # Column scaling invariance: rescale column 1 by 1000
    scale = np.array([1.0, 1000.0, 1.0])
    cov_orig = prewhitened_hac_covariance(X, u, kernel="bartlett")
    cov_scaled = prewhitened_hac_covariance(X * scale[None, :], u, kernel="bartlett")
    expected = (cov_orig / scale[:, None]) / scale[None, :]
    np.testing.assert_allclose(cov_scaled, expected, rtol=1e-6, atol=1e-10)


def test_cr2_thin_svd_matches_exact_dense_formula() -> None:
    rng = np.random.default_rng(606)
    G, m = 12, 20
    g = np.repeat(np.arange(G), m)
    X = np.column_stack([np.ones(G * m), rng.normal(0, 1, G * m), (g >= 6).astype(float)])
    y = X @ np.array([1.0, 2.0, -0.5]) + rng.normal(0, 1, G * m)
    fit = least_squares(X, y)
    res = cr2_covariance(X, fit.residuals, g)

    # Verify against explicit dense (I - H_gg)^(-1/2)
    bread = np.linalg.inv(X.T @ X)
    scores = np.zeros((G, X.shape[1]))
    for j in range(G):
        idx = np.flatnonzero(g == j)
        xg, ug = X[idx], fit.residuals[idx]
        h_gg = xg @ bread @ xg.T
        ev, evec = np.linalg.eigh(np.eye(len(idx)) - h_gg)
        ag = (evec * (ev ** -0.5)) @ evec.T
        scores[j] = xg.T @ (ag @ ug)
    cov_dense = bread @ (scores.T @ scores) @ bread
    np.testing.assert_allclose(res.covariance, cov_dense, rtol=1e-10, atol=1e-12)


def test_honest_did_varying_vs_universal_and_smoothness_covariance() -> None:
    # Universal event study: beta = [-0.2, -0.1, 0.0 (ref=-1), 1.5, 1.8]
    et = [-3, -2, -1, 0, 1]
    beta_univ = np.array([-0.2, -0.1, 0.0, 1.5, 1.8])
    # Varying pre-diffs: e=-3 is beta[-3] - (-0.3)? For [-3,-2] relative to adjacent:
    # beta_univ[-2] = -d[-2] => d[-2] = +0.1; beta_univ[-3] = -(d[-3] + d[-2]) => d[-3] = +0.1
    beta_var = np.array([0.1, 0.1, 0.0, 1.5, 1.8])
    cov_u = np.diag([0.04, 0.04, 0.0, 0.09, 0.09])
    # Transform cov_u to varying pre-diffs: d = A_inv @ beta_univ
    a_inv = np.eye(5)
    a_inv[0, 0], a_inv[0, 1] = -1.0, 1.0
    a_inv[1, 1] = -1.0
    cov_v = a_inv @ cov_u @ a_inv.T

    res_u = honest_did_sensitivity(et, beta_univ, cov_u, base_period="universal", method="smoothness", m_grid=[0.0, 0.1])
    res_v = honest_did_sensitivity(et, beta_var, cov_v, base_period="varying", method="smoothness", m_grid=[0.0, 0.1])
    assert res_u.target_se == pytest.approx(res_v.target_se, abs=1e-10)
    # Smoothness SE must strictly exceed post-only SE (sqrt(0.09) = 0.30) because pre-slope is noisy!
    assert res_u.target_se > 0.30 + 0.01
    pd.testing.assert_frame_equal(res_u.table, res_v.table)


def test_audit_panel_nonbinary_and_imbalanced_rollout() -> None:
    # Non-binary strings must return is_viable=False rather than crashing
    bad_df = pd.DataFrame({"u": [1, 1, 2, 2], "t": [1, 2, 1, 2], "d": ["no", "yes", "no", "no"]})
    audit_bad = audit_panel(bad_df, "u", "t", "d")
    assert not audit_bad.is_viable
    assert any("non-binary" in b.lower() for b in audit_bad.blocker_reasons)

    # Imbalanced panel where both treated units adopt at time=2020 (different positional indices!)
    imb_df = pd.DataFrame({
        "u": [1, 1, 1, 2, 2, 3, 3, 3],
        "t": [2018, 2019, 2020, 2019, 2020, 2018, 2019, 2020],
        "d": [0, 0, 1, 0, 1, 0, 0, 0],
    })
    audit_imb = audit_panel(imb_df, "u", "t", "d")
    assert audit_imb.treatment_type == "single_shot"


def test_ai_security_and_reporting_options_immutability() -> None:
    with pytest.raises(ValueError, match="Unknown AI provider"):
        quasi.ai.configure(provider="nonexistent_provider")  # type: ignore[arg-type]

    from quasi.ai.adapters import OpenAIAdapter
    bad_adapter = OpenAIAdapter(api_key="k", base_url="file:///etc/passwd")
    with pytest.raises(ValueError, match="http:// or https://"):
        bad_adapter.generate("sys", "usr")

    # TableFormattingOptions must not be mutated in place for style='nature'
    rng = np.random.default_rng(707)
    df = pd.DataFrame({"t": np.arange(40), "y": 5.0 + 0.1 * np.arange(40) + 2.0 * (np.arange(40) >= 20) + rng.normal(0, 0.5, 40)})
    r = quasi.InterruptedTimeSeries(df, "y", "t", 20).fit()
    opts = TableFormattingOptions(style="nature", se_type="parens", stars={0.05: "**"})
    tbl = ModelComparisonTable({"ITS": r}, options=opts)
    assert opts.se_type == "parens" and opts.stars == {0.05: "**"}
    assert "Notes" in tbl.to_latex()

    # New DesignSpec fields route cleanly
    rd_df = pd.DataFrame({"x": rng.uniform(-1, 1, 250)})
    rd_df["y"] = 1.0 + 0.5 * rd_df["x"] + 2.0 * (rd_df["x"] >= 0) + rng.normal(0, 0.3, 250)
    rd_res = quasi.estimate(rd_df, design=quasi.RDDDesign(outcome="y", running_var="x", cutoff=0.0, bandwidth="mserd", vce="nn"))
    assert "nn" in rd_res.inference_details["method"]

    # MatchingResult point-only metadata serializes None (not NaN)
    obs_df = pd.DataFrame({"x": rng.normal(0, 1, 120)})
    obs_df["d"] = rng.binomial(1, 0.5, 120)
    obs_df["y"] = 1.0 + obs_df["x"] + 2.0 * obs_df["d"] + rng.normal(0, 0.5, 120)
    psm_res = PropensityScoreMatching(obs_df, "y", "d", ["x"]).fit()
    meta = psm_res.to_metadata_dict()
    assert meta["primary_metric"]["se"] is None
    assert meta["primary_metric"]["ci"] == (None, None)

    # Clustered IV refuses silent homoskedastic CLR
    iv_df = pd.DataFrame({
        "g": np.repeat(np.arange(15), 10),
        "z": rng.normal(0, 1, 150),
    })
    iv_df["d"] = 0.8 * iv_df["z"] + rng.normal(0, 1, 150)
    iv_df["y"] = 2.0 * iv_df["d"] + rng.normal(0, 1, 150)
    iv_cl = quasi.TwoStageLS(iv_df, "y", "d", ["z"], cluster="g").fit()
    with pytest.raises(UnsupportedInferenceError):
        iv_cl.clr_test()
