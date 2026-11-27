"""Tests for Phase 2B frontier shared inference engine (quasi.inference)."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.inference import (
    cluster_covariance,
    cr2_covariance,
    ewc_har_covariance,
    hac_covariance,
    least_squares,
    multiway_cluster_covariance,
    prewhitened_hac_covariance,
    profile_ml_ar1,
    reml_ar1,
    solve_simplex_regression,
    two_stage_least_squares,
    wild_cluster_bootstrap,
)


def test_weighted_ols_matches_statsmodels_wls():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(101)
    n = 150
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    y = x @ np.array([1.0, 2.5, -0.8]) + rng.normal(size=n)
    w = rng.uniform(0.5, 5.0, size=n)

    fit = least_squares(x, y, weights=w)
    ref = sm.WLS(y, x, weights=w).fit()
    assert_allclose(fit.coefficients, ref.params, atol=1e-11, rtol=1e-10)
    assert_allclose(fit.residuals, y - x @ ref.params, atol=1e-11)

    with pytest.raises(InvalidDesignError):
        least_squares(x, y, weights=np.zeros(n))


def test_weighted_2sls_exact_equivalence_to_scaled_2sls():
    rng = np.random.default_rng(102)
    n = 200
    z_e = rng.normal(size=(n, 2))
    u = rng.normal(size=n)
    x_e = 0.7 * z_e[:, 0] + 0.5 * z_e[:, 1] + 0.6 * u + rng.normal(size=n)
    y = 1.0 + 1.8 * x_e + 0.6 * u + rng.normal(size=n)
    w = rng.uniform(0.2, 4.0, size=n)

    x_mat = np.column_stack([np.ones(n), x_e])
    z_mat = np.column_stack([np.ones(n), z_e])
    fit_w, _ = two_stage_least_squares(x_mat, z_mat, y, weights=w)
    sw = np.sqrt(w)
    fit_scaled, _ = two_stage_least_squares(x_mat * sw[:, None], z_mat * sw[:, None], y * sw)
    assert_allclose(fit_w.coefficients, fit_scaled.coefficients, atol=1e-11, rtol=1e-10)


def test_cr2_and_satterthwaite_match_bruteforce_bell_mccaffrey():
    rng = np.random.default_rng(103)
    g = 12
    n_per = 10
    n = g * n_per
    groups = np.repeat(np.arange(g), n_per)
    trt = (groups < g // 2).astype(float)
    x = np.column_stack([np.ones(n), trt, rng.normal(size=n)])
    y = 2.0 + 1.5 * trt + 0.4 * x[:, 2] + rng.normal(size=n)
    fit = least_squares(x, y)

    cr2 = cr2_covariance(x, fit.residuals, groups)
    assert cr2.n_clusters == g
    assert cr2.satterthwaite_df.shape == (3,)

    # Independent brute-force N x N verification of Bell-McCaffrey (2002) eq. (10)
    bread = np.linalg.inv(x.T @ x)
    h_full = x @ bread @ x.T
    m_full = np.eye(n) - h_full
    a_blocks = []
    for cl in range(g):
        idx = np.flatnonzero(groups == cl)
        eigv, eigvec = np.linalg.eigh(m_full[np.ix_(idx, idx)])
        a_blocks.append((idx, (eigvec * (eigv ** -0.5)) @ eigvec.T))

    # Check CR2 covariance matrix
    scores = np.zeros((g, 3))
    for cl, (idx, a_g) in enumerate(a_blocks):
        scores[cl] = x[idx].T @ a_g @ fit.residuals[idx]
    v_ref = bread @ (scores.T @ scores) @ bread
    assert_allclose(cr2.covariance, v_ref, atol=1e-11, rtol=1e-10)

    # Check Satterthwaite df for each parameter
    for k in range(3):
        c = np.zeros(3)
        c[k] = 1.0
        a_vecs = []
        for idx, a_g in a_blocks:
            # a_g_vec = M_{., idx} @ A_g @ X_g @ B @ c
            a_vecs.append(m_full[:, idx] @ a_g @ x[idx] @ bread @ c)
        a_mat = np.column_stack(a_vecs)  # N x G
        p_ref = a_mat.T @ a_mat
        df_ref = (np.trace(p_ref) ** 2) / np.sum(p_ref * p_ref)
        assert cr2.satterthwaite_df[k] == pytest.approx(df_ref, rel=1e-10, abs=1e-10)


def test_multiway_cluster_covariance_matches_inclusion_exclusion():
    rng = np.random.default_rng(104)
    n_firms, n_years = 20, 15
    n = n_firms * n_years
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    u = rng.normal(size=n) + rng.normal(size=n_firms)[firm] + rng.normal(size=n_years)[year]
    y = x @ np.array([1.0, 2.0]) + u
    fit = least_squares(x, y)

    mw = multiway_cluster_covariance(x, fit.residuals, [firm, year], small_sample=False, enforce_psd=False)
    v_f = cluster_covariance(x, fit.residuals, firm, small_sample=False)
    v_y = cluster_covariance(x, fit.residuals, year, small_sample=False)
    inter = firm * 1000 + year
    v_fy = cluster_covariance(x, fit.residuals, inter, small_sample=False)
    assert_allclose(mw.covariance, v_f + v_y - v_fy, atol=1e-11, rtol=1e-10)
    assert mw.cluster_counts == (n_firms, n_years)


def test_ewc_har_achieves_strong_coverage_under_high_ar1():
    # Direct finite-sample comparison: EWC fixed-b HAR vs default Bartlett NW-HAC at rho=0.80
    rng = np.random.default_rng(2026)
    n = 80
    reps = 180
    t_seq = np.arange(n, dtype=float)
    d_post = (t_seq >= n // 2).astype(float)
    x = np.column_stack([np.ones(n), t_seq, d_post])
    cov_ewc_hits = 0
    cov_hac_hits = 0
    for _ in range(reps):
        eps = np.zeros(n)
        u = rng.normal(size=n)
        eps[0] = u[0] / np.sqrt(1.0 - 0.8**2)
        for i in range(1, n):
            eps[i] = 0.8 * eps[i - 1] + u[i]
        y = 5.0 + 0.1 * t_seq + 2.0 * d_post + eps
        fit = least_squares(x, y)
        ewc = ewc_har_covariance(x, fit.residuals)
        se_ewc = float(np.sqrt(ewc.covariance[2, 2]))
        crit_ewc = float(stats.t.ppf(0.975, df=ewc.degrees_freedom))
        cov_ewc_hits += int(abs(fit.coefficients[2] - 2.0) <= crit_ewc * se_ewc)

        v_hac = hac_covariance(x, fit.residuals)
        se_hac = float(np.sqrt(v_hac[2, 2]))
        cov_hac_hits += int(abs(fit.coefficients[2] - 2.0) <= 1.96 * se_hac)

    rate_ewc = cov_ewc_hits / reps
    rate_hac = cov_hac_hits / reps
    # EWC fixed-b HAR must substantially outperform naive NW-HAC under strong AR(1)
    assert rate_ewc > rate_hac + 0.10
    assert rate_ewc >= 0.88


def test_prewhitened_hac_bartlett_and_qs():
    rng = np.random.default_rng(105)
    n = 120
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.7 * eps[i - 1] + rng.normal()
    y = x @ np.array([1.0, 0.5]) + eps
    fit = least_squares(x, y)
    v_b = prewhitened_hac_covariance(x, fit.residuals, kernel="bartlett")
    v_qs = prewhitened_hac_covariance(x, fit.residuals, kernel="qs")
    assert np.all(np.linalg.eigvalsh(v_b) > 0)
    assert np.all(np.linalg.eigvalsh(v_qs) > 0)


def test_wild_cluster_bootstrap_rademacher_and_webb():
    rng = np.random.default_rng(106)
    g = 8  # triggers auto -> webb
    n = g * 20
    groups = np.repeat(np.arange(g), 20)
    trt = (groups < g // 2).astype(float)
    x = np.column_stack([np.ones(n), trt])
    cl_err = rng.normal(size=g)[groups]
    y_null = 2.0 + cl_err + rng.normal(scale=0.5, size=n)
    y_alt = y_null + 3.0 * trt

    res_null = wild_cluster_bootstrap(x, y_null, groups, 1, null_value=0.0, reps=199, seed=7)
    res_alt = wild_cluster_bootstrap(x, y_alt, groups, 1, null_value=0.0, reps=199, seed=7)
    assert res_null.weight_type == "webb"
    assert res_alt.p_value < 0.05
    assert 0.0 < res_null.p_value <= 1.0

    with pytest.raises(InsufficientSupportError):
        wild_cluster_bootstrap(x, y_null, groups % 3, 1, reps=99)


def test_reml_ar1_reduces_hurwicz_bias_vs_profile_ml():
    rng = np.random.default_rng(107)
    n = 50
    t = np.arange(n, dtype=float)
    x = np.column_stack([np.ones(n), t, (t >= 25).astype(float), np.maximum(0.0, t - 25)])
    rhos_ml, rhos_reml = [], []
    for _ in range(30):
        eps = np.zeros(n)
        u = rng.normal(size=n)
        eps[0] = u[0] / np.sqrt(1 - 0.7**2)
        for i in range(1, n):
            eps[i] = 0.7 * eps[i - 1] + u[i]
        y = 10.0 + 0.1 * t + 2.0 * (t >= 25) + eps
        rhos_ml.append(profile_ml_ar1(x, y).rho)
        rhos_reml.append(reml_ar1(x, y).rho)
    # REML should have strictly less downward bias (mean closer to true 0.7) than profile ML
    assert np.mean(rhos_reml) > np.mean(rhos_ml)


def test_shared_simplex_regression_solver():
    rng = np.random.default_rng(108)
    a = rng.normal(size=(25, 6))
    w_true = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0])
    b = 3.5 + a @ w_true
    sol = solve_simplex_regression(a, b, ridge=0.0, intercept=True)
    assert sol.converged
    assert sol.intercept == pytest.approx(3.5, abs=1e-4)
    assert_allclose(sol.weights, w_true, atol=1e-3)
