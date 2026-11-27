"""Phase 5 numerical contracts. Comparing algebra to Statsmodels is NOT causal validation."""
import json

import numpy as np
import pandas as pd
import pytest

from quasi.errors import InvalidDesignError, InsufficientSupportError, NotIdentifiedError
from quasi.inference import (least_squares, two_stage_least_squares, hc_covariance,
    hac_covariance, cluster_covariance, bootstrap_indices, percentile_interval)


def example(seed=7754, n=180):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z + 0.4 * w + v
    y = 1.6 * x + 0.7 * w + 0.8 * v + rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, w])
    Z = np.column_stack([np.ones(n), z, w])
    return X, Z, y


def test_rank_safe_least_squares_matches_numpy_and_invariance():
    X, _, y = example()
    got = least_squares(X, y)
    np.testing.assert_allclose(got.coefficients, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-12)
    np.testing.assert_allclose(least_squares(X, y + 10).coefficients,
        got.coefficients + [10, 0, 0], atol=1e-12)


def test_iv_2sls_matches_statsmodels_coefficient_and_has_structural_residual():
    sm = pytest.importorskip('statsmodels')
    from statsmodels.sandbox.regression.gmm import IV2SLS
    X, Z, y = example()
    model, projected = two_stage_least_squares(X, Z, y)
    ref = IV2SLS(y, X, Z).fit()
    np.testing.assert_allclose(model.coefficients, ref.params, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(model.residuals, y - X @ ref.params, atol=1e-10)
    np.testing.assert_allclose(projected, Z @ np.linalg.lstsq(Z, X, rcond=None)[0], atol=1e-10)


def test_iv_quasi_matches_independent_statsmodels_across_controls():
    import quasi
    from statsmodels.sandbox.regression.gmm import IV2SLS
    X, Z, y = example()
    data = pd.DataFrame({'x': X[:,1], 'z':Z[:,1], 'w':X[:,2], 'y':y})
    result = quasi.TwoStageLS(data, 'y', 'x', ['z'], covariates=['w']).fit()
    ref = IV2SLS(y, X, Z).fit()
    assert result.estimate == pytest.approx(ref.params[1], rel=1e-9, abs=1e-10)
    assert result.se > 0
    record = result.to_record()
    assert record.inference.method == 'IV_2SLS_HC0'
    assert record.inference.independently_validated is False
    assert record.to_dict()['additional_metadata']['inference_settings']['covariance'] == 'HC0'
    assert json.loads(record.to_json())['experimental'] is True


def test_hc_implementations_match_statsmodels_for_all_variants():
    sm = pytest.importorskip('statsmodels.api')
    X, _, y = example()
    ref = sm.OLS(y, X).fit()
    for kind in ('HC0', 'HC1', 'HC2', 'HC3'):
        np.testing.assert_allclose(hc_covariance(X, ref.resid, kind),
                                   ref.get_robustcov_results(cov_type=kind).cov_params(),
                                   rtol=2e-10, atol=1e-10)


@pytest.mark.parametrize('lag', [0, 1, 3, 12])
def test_hac_matches_statsmodels_and_zero_lag_is_hc0(lag):
    sm = pytest.importorskip('statsmodels.api')
    X, _, y = example()
    ref = sm.OLS(y, X).fit()
    desired = ref.get_robustcov_results(cov_type='HAC', maxlags=lag,
                                        use_correction=False).cov_params()
    np.testing.assert_allclose(hac_covariance(X, ref.resid, max_lag=lag),
                               desired, rtol=2e-10, atol=1e-10)
    if lag == 0:
        np.testing.assert_allclose(hac_covariance(X, ref.resid, max_lag=lag),
                                   hc_covariance(X, ref.resid, 'HC0'))


def test_cluster_cr0_cr1_parity_statsmodels():
    sm = pytest.importorskip('statsmodels.api')
    X, _, y = example()
    groups = np.repeat(np.arange(30), 6)
    ref = sm.OLS(y, X).fit()
    for correction in [True, False]:
        desired = ref.get_robustcov_results(
            cov_type='cluster', groups=groups, use_correction=correction).cov_params()
        np.testing.assert_allclose(cluster_covariance(X, ref.resid, groups,
                                                       small_sample=correction), desired,
                                   rtol=5e-10, atol=1e-10)


def test_rank_pathologies_fail_closed_no_pseudoinverse():
    X, Z, y = example()
    bad = X.copy()
    bad[:,2] = 2 * bad[:,1]
    with pytest.raises(NotIdentifiedError, match='rank-deficient'):
        least_squares(bad, y)
    with pytest.raises(NotIdentifiedError):
        two_stage_least_squares(X, np.column_stack([Z[:,0],Z[:,1],Z[:,1]]), y)
    with pytest.raises(NotIdentifiedError):
        hc_covariance(bad, y)
    with pytest.raises(InvalidDesignError, match='finite'):
        least_squares(X * float('nan'), y)


def test_covariance_specification_rejects_unsafe_conditions():
    X, _, y = example()
    resid = least_squares(X, y).residuals
    with pytest.raises(InvalidDesignError):
        hac_covariance(X, resid, max_lag=-1)
    with pytest.raises(InvalidDesignError):
        hac_covariance(X, resid, max_lag=len(X))
    with pytest.raises(InvalidDesignError):
        hac_covariance(X, resid, max_lag=True)
    with pytest.raises(InvalidDesignError):
        hc_covariance(X, resid, 'foo')
    with pytest.raises(InsufficientSupportError):
        cluster_covariance(X, resid, np.ones(len(y)))


def test_bootstrap_seed_cluster_boundaries_and_percentile():
    groups = np.repeat(np.arange(4), [2, 5, 3, 4])
    draws = bootstrap_indices(len(groups), reps=30, seed=11, groups=groups)
    other = bootstrap_indices(len(groups), reps=30, seed=11, groups=groups)
    assert len(draws) == 30
    assert all(np.array_equal(a, b) for a, b in zip(draws, other))
    assert any(not np.array_equal(a, b) for a, b in zip(draws, bootstrap_indices(len(groups), reps=30, seed=12, groups=groups)))
    for draw in draws:
        # each sampled cluster is fully represented in each sampled block
        i = 0
        while i < len(draw):
            size = int(np.sum(groups == groups[draw[i]]))
            np.testing.assert_array_equal(draw[i:i + size], np.flatnonzero(groups == groups[draw[i]]))
            i += size
    interval = percentile_interval(np.arange(1.0, 101.0), alpha=0.10)
    assert interval == pytest.approx((5.95, 95.05))
    with pytest.raises(InvalidDesignError):
        percentile_interval(np.array([1.0, np.nan]))
    with pytest.raises(InsufficientSupportError):
        bootstrap_indices(3, reps=2, seed=1, groups=np.ones(3))


def test_its_report_records_actual_hac_lag_and_respects_explicit_zero():
    import quasi
    n = 80
    rng = np.random.default_rng(7)
    t = np.arange(n)
    y = 5.0 + 0.04 * t + 1.2 * (t >= 40) + 0.1 * rng.normal(size=n)
    frame = pd.DataFrame({'y':y, 't':t})
    result = quasi.InterruptedTimeSeries(frame, 'y', 't', 40, hac_lag=0).fit()
    assert result.inference_details['max_lag'] == 0
    assert result.to_record().inference.method == 'OLS_HAC_Bartlett'
    assert result.to_record().to_dict()['additional_metadata']['inference_settings']['max_lag'] == 0
    with pytest.raises(InvalidDesignError):
        quasi.InterruptedTimeSeries(frame, 'y', 't', 40, hac_lag=80).fit()


def test_checked_simplex_solver_kkt_and_reference_optimizer():
    from scipy.optimize import minimize
    from quasi.inference import checked_simplex_qp
    rng = np.random.default_rng(12)
    a = rng.normal(size=(14, 6))
    q = a.T @ a + 0.2 * np.eye(6)
    p = rng.normal(size=6)
    result = checked_simplex_qp(q, p, tol=1e-10)
    reference = minimize(lambda w: 0.5*w @ q @ w - p @ w,
        np.ones(6)/6, jac=lambda w: q @ w - p,
        constraints=[{'type':'eq','fun': lambda w: np.sum(w)-1, 'jac':lambda w: np.ones(6)}],
        bounds=[(0, 1)]*6, method='SLSQP', options={'ftol':1e-12, 'maxiter':5000})
    assert reference.success, reference.message
    assert result.converged and result.projected_stationarity <= 1e-10
    assert abs(sum(result.weights)-1) < 1e-12
    assert np.min(result.weights) >= -1e-14
    assert result.objective == pytest.approx(reference.fun, abs=1e-8)


def test_checked_simplex_solver_blocks_nonconvergence_and_nonconvexity():
    from quasi.inference import checked_simplex_qp
    from quasi.errors import ConvergenceError
    with pytest.raises(InvalidDesignError, match='convex'):
        checked_simplex_qp(np.diag([-2., 1.]), np.zeros(2))
    with pytest.raises(InvalidDesignError, match='finite'):
        checked_simplex_qp(np.eye(2), np.array([float('nan'), 1.]))
    with pytest.raises(ConvergenceError):
        checked_simplex_qp(np.diag([1., 100.]), np.array([0., 9.]), tol=1e-15, max_iter=1)


def test_large_iv_fit_avoids_dense_projection_n_squared():
    from quasi.inference import two_stage_least_squares
    rng = np.random.default_rng(85)
    n = 15000
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = 1 + 1.4*x + rng.normal(size=n)
    design = np.column_stack([np.ones(n), x])
    instrument = np.column_stack([np.ones(n), z])
    fit, projected = two_stage_least_squares(design, instrument, y)
    assert projected.shape == (n, 2)
    assert fit.coefficients[1] == pytest.approx(1.4, abs=0.08)


def test_its_lag_sensitivity_reproduces_deterministic_paired_series():
    from validation.its_lag_sensitivity import run
    a = run(reps=3, seed=97, lags=(0,3))
    b = run(reps=3, seed=97, lags=(0,3))
    assert a == b
    assert len(a) == 4
    assert all(entry.inference_status == 'exploratory_unvalidated' for entry in a)
