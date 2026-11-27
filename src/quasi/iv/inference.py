"""Experimental IV inference and conditional-instrument relevance diagnostics.

All hypothesis tests are for a *single* endogenous regressor; exogenous controls
are included in both the structural equation and the instrument span. There is
no implicit assumption that an F-statistic certifies instrument strength.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats

from quasi.errors import InvalidDesignError, InsufficientSupportError, NotIdentifiedError
from quasi.inference import least_squares, hc_covariance, cluster_covariance
from quasi.inference.covariance import _validate_groups


@dataclass(frozen=True)
class WaldTest:
    statistic: float
    p_value: float
    distribution: str
    df_numerator: int
    df_denominator: int | None = None


def covariance_for_iv(x: np.ndarray, residuals: np.ndarray, *, kind: str,
                      groups: np.ndarray | None = None) -> np.ndarray:
    """Score covariance. For IV's main equation use *projected* X and structural u."""
    if kind in ('HC0', 'HC1') and groups is None:
        return hc_covariance(x, residuals, kind=kind)
    if kind in ('CR0', 'CR1') and groups is not None:
        return cluster_covariance(x, residuals, groups, small_sample=kind == 'CR1')
    raise InvalidDesignError('IV covariance must be HC0/HC1 without clusters or CR0/CR1 with clusters')


def _test_zero(coefficients: np.ndarray, covariance: np.ndarray, *,
               groups: np.ndarray | None = None) -> WaldTest:
    """Wald test of q coefficients: chi-square for iid, F(q,G-1) for clusters."""
    b = np.asarray(coefficients, dtype=float).reshape(-1)
    v = np.asarray(covariance, dtype=float)
    if v.shape != (len(b), len(b)) or not np.isfinite(v).all():
        raise InvalidDesignError('Malformed hypothesis covariance')
    try:
        eigenvalues = np.linalg.eigvalsh(v)
        if eigenvalues[0] <= max(1e-14, 1e-10 * eigenvalues[-1]):
            raise InsufficientSupportError('Hypothesis covariance is singular; inference unavailable')
        w = max(0.0, float(b @ np.linalg.solve(v, b)))
    except np.linalg.LinAlgError as exc:
        raise InsufficientSupportError('Hypothesis covariance cannot be inverted') from exc
    if groups is None:
        return WaldTest(w, float(stats.chi2.sf(w, len(b))), 'chi2', len(b))
    g = len(np.unique(groups))
    f = w / len(b)
    return WaldTest(f, float(stats.f.sf(f, len(b), g-1)), 'F', len(b), g-1)


def excluded_instruments_test(first_stage_coefficients: np.ndarray,
                              first_stage_covariance: np.ndarray, q: int,
                              groups: np.ndarray | None = None) -> WaldTest:
    """Joint excluded-instrument significance in unrestricted first stage.

    The coefficient layout is [constant, excluded Z (q columns), controls W].
    The result is a robust/cluster Wald test, NOT a Stock-Yogo effective F.
    """
    return _test_zero(first_stage_coefficients[1:1+q],
                      first_stage_covariance[1:1+q, 1:1+q], groups=groups)


def anderson_rubin_test(y: np.ndarray, x_end: np.ndarray, z_full: np.ndarray,
                        q: int, null_effect: float, *, kind: str,
                        groups: np.ndarray | None = None) -> WaldTest:
    """H0: beta_endogenous = null_effect by Wald of excluded Z in Y-beta0*X on Z,W.

    Single endogenous regressor, arbitrary number of excluded instruments.
    Heteroskedastic chi-square and cluster F approximations; finite-cluster
    exactness and nuisance-endogeneity beyond the single X are NOT claimed.
    """
    if isinstance(null_effect, bool) or not np.isscalar(null_effect) or not np.isfinite(null_effect):
        raise InvalidDesignError('AR null_effect must be finite scalar')
    fit = least_squares(z_full, y - float(null_effect) * x_end)
    v = covariance_for_iv(z_full, fit.residuals, kind=kind, groups=groups)
    return excluded_instruments_test(fit.coefficients, v, q, groups)


@dataclass(frozen=True)
class ARConfidenceSet:
    """Experimental analytic inversion of a one-instrument robust AR Wald test.

    The accepted set is a union of closed intervals. Infinite endpoints are
    represented with +/-np.inf. This is a confidence *set*, not necessarily a
    bounded interval. It is conditional on the excluded instrument being valid;
    asymptotic HC/cluster reference distributions are not finite-sample exact.
    """
    intervals: tuple[tuple[float, float], ...]
    level: float
    covariance: str
    distribution: str

    @property
    def is_empty(self) -> bool:
        return not self.intervals

    @property
    def is_unbounded(self) -> bool:
        return any(not np.isfinite(lo) or not np.isfinite(hi)
                   for lo, hi in self.intervals)

    @property
    def is_disconnected(self) -> bool:
        return len(self.intervals) > 1

    def contains(self, effect: float) -> bool:
        if not np.isscalar(effect) or isinstance(effect, bool) or not np.isfinite(effect):
            raise InvalidDesignError('AR set membership requires a finite numeric effect')
        return any(lo <= effect <= hi for lo, hi in self.intervals)


def _quadratic_nonpositive(a: float, b: float, c: float) -> tuple[tuple[float, float], ...]:
    """All real t satisfying a*t**2+b*t+c <= 0; never clip to a grid."""
    if not np.isfinite([a, b, c]).all():
        raise InsufficientSupportError('Nonfinite AR inversion polynomial')
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0:
        return ((-np.inf, np.inf),)
    a, b, c = a/scale, b/scale, c/scale
    tol = 2e-13
    if abs(a) <= tol:
        if abs(b) <= tol:
            return ((-np.inf, np.inf),) if c <= 0 else ()
        root = -c/b
        return ((-np.inf, root),) if b > 0 else ((root, np.inf),)
    disc = b*b - 4*a*c
    if disc < -tol:
        return ((-np.inf, np.inf),) if a < 0 else ()
    if abs(disc) <= tol:
        root = -b/(2*a)
        return ((-np.inf, np.inf),) if a < 0 else ((root, root),)
    sqrt_d = float(np.sqrt(disc))
    # Stable quadratic formula avoids cancellation in one root.
    q = -0.5*(b + np.copysign(sqrt_d, b))
    roots = sorted((q/a, c/q)) if q else sorted(((-b-sqrt_d)/(2*a), (-b+sqrt_d)/(2*a)))
    low, high = map(float, roots)
    if a > 0:
        return ((low, high),)
    return ((-np.inf, low), (high, np.inf))


def anderson_rubin_confidence_set(y: np.ndarray, x_end: np.ndarray,
                                  z_full: np.ndarray, q: int, *,
                                  alpha: float = 0.05, kind: str = 'HC0',
                                  groups: np.ndarray | None = None) -> ARConfidenceSet:
    """Analytically invert one-excluded-instrument robust Anderson--Rubin test.

    A single endogenous regressor and a single excluded instrument are required.
    The reduced form residual and coefficient are affine in the candidate beta,
    while its HC/cluster variance is quadratic; inversion thus solves a global
    quadratic inequality with no arbitrary effect bounds or grid truncation.

    Multiple excluded instruments require a distinct matrix-valued inversion
    method and are deliberately rejected rather than quietly grid-truncated.
    """
    if q != 1:
        raise InvalidDesignError('Analytic AR confidence set currently requires exactly one excluded instrument')
    if isinstance(alpha, bool) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise InvalidDesignError('alpha must be a finite probability strictly between 0 and 1')
    z = np.asarray(z_full, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x_end, dtype=float).reshape(-1)
    if z.ndim != 2 or z.shape[0] != len(y) or len(x) != len(y) or z.shape[1] < 2:
        raise InvalidDesignError('AR design matrix must contain constant, excluded instrument, optional controls')
    # OLS routines reject rank deficiency and nonfinite input.
    fit_y, fit_x = least_squares(z, y), least_squares(z, x)
    n, p = z.shape
    qmat, r = np.linalg.qr(z, mode='reduced')
    h = qmat @ np.linalg.solve(r.T, np.eye(p)[1])
    uy, ux = fit_y.residuals, fit_x.residuals
    ay, ax = h*uy, h*ux
    if kind in ('HC0', 'HC1') and groups is None:
        factor = 1.0 if kind == 'HC0' else n/(n-p)
        cutoff = float(stats.chi2.ppf(1-alpha, 1))
        dist = 'chi2(1)'
    elif kind in ('CR0', 'CR1') and groups is not None:
        labels = np.asarray(groups)
        if labels.shape != (n,):
            raise InvalidDesignError('AR cluster labels must be one per observation')
        try:
            import pandas as pd
            if pd.isna(labels).any():
                raise InvalidDesignError('AR cluster labels cannot be missing')
            _, idx = np.unique(labels, return_inverse=True)
        except (TypeError, ValueError) as exc:
            raise InvalidDesignError('AR cluster labels must be comparable and nonmissing') from exc
        g = int(idx.max()+1)
        if g < 10:
            raise InsufficientSupportError('Clustered IV confidence sets require at least 10 groups')
        ayc, axc = np.zeros(g), np.zeros(g)
        np.add.at(ayc, idx, ay)
        np.add.at(axc, idx, ax)
        ay, ax = ayc, axc
        factor = 1.0 if kind == 'CR0' else (g/(g-1))*((n-1)/(n-p))
        cutoff = float(stats.f.ppf(1-alpha, 1, g-1))
        dist = f'F(1,{g-1})'
    else:
        raise InvalidDesignError('AR covariance must be HC0/HC1 unclustered or CR0/CR1 clustered')
    syy, sxx, sxy = factor*float(ay@ay), factor*float(ax@ax), factor*float(ay@ax)
    # If the variance polynomial touches zero, there is a null for which the
    # inversion statistic is undefined. Do not claim a globally valid set.
    if sxx <= 0 or syy <= 0 or (syy*sxx-sxy*sxy) <= 1e-13*syy*sxx:
        raise InsufficientSupportError('AR residual-score variance vanishes for some null; inversion unavailable')
    by, bx = float(fit_y.coefficients[1]), float(fit_x.coefficients[1])
    polynomial = (bx*bx-cutoff*sxx,
                  -2*(by*bx-cutoff*sxy),
                  by*by-cutoff*syy)
    return ARConfidenceSet(_quadratic_nonpositive(*polynomial),
                           float(1-alpha), kind, dist)


@dataclass(frozen=True)
class GMMOveridentificationTest:
    """Auxiliary two-step, unclustered heteroskedastic GMM Hansen J result.

    Not a test of the validity of any particular instrument; non-rejection
    does not verify exclusion. Coefficients belong to the auxiliary GMM fit,
    not to the primary 2SLS estimator.
    """
    statistic: float
    p_value: float
    degrees_of_freedom: int
    gmm_coefficients: tuple[float, ...]
    method: str = 'auxiliary_two_step_GMM_HC0_uncentered'


def gmm_hansen_j(
    y: np.ndarray,
    structural_x: np.ndarray,
    full_z: np.ndarray,
    *,
    groups: np.ndarray | None = None,
) -> GMMOveridentificationTest:
    """Two-step heteroskedastic or one-way clustered GMM J for overidentified linear IV.

    1. Consistent 2SLS preliminary beta.
    2. Optimal moment weight from S0 = n^-1 sum_i (z_i u0_i)(z_i u0_i)' (or
       whole-cluster sum S0 = n^-1 sum_g (sum_{i in g} z_i u0_i)(...)' if clustered).
    3. Refit efficient two-step GMM using the *fixed* first-step weight.
    4. Test n * gbar(beta_GMM)' S0^-1 gbar(beta_GMM), chi2(l - k).
    """
    from quasi.inference import two_stage_least_squares
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(structural_x, dtype=float)
    z = np.asarray(full_z, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[0] != len(y) or z.shape[0] != len(y):
        raise InvalidDesignError('GMM J requires aligned two-dimensional structural and instrument matrices')
    n, k = x.shape
    l = z.shape[1]
    if l <= k:
        raise InvalidDesignError('Hansen J requires more instruments (including controls) than structural regressors')
    preliminary, _ = two_stage_least_squares(x, z, y)
    u = preliminary.residuals
    scores = z * u[:, None]
    if groups is None:
        moment_cov = (scores.T @ scores) / n
        method_label = 'auxiliary_two_step_GMM_HC0_uncentered'
    else:
        labels = np.asarray(groups)
        if labels.ndim != 1 or len(labels) != n:
            raise InvalidDesignError('GMM J cluster labels must align with observations')
        _, inv = np.unique(labels, return_inverse=True)
        g_cnt = int(inv.max() + 1)
        if g_cnt < 10:
            raise InsufficientSupportError('Clustered GMM Hansen J requires at least 10 clusters')
        c_scores = np.zeros((g_cnt, l), dtype=float)
        np.add.at(c_scores, inv, scores)
        moment_cov = (c_scores.T @ c_scores) / n
        method_label = 'auxiliary_two_step_GMM_CR0_uncentered'
    evals = np.linalg.eigvalsh(moment_cov)
    if evals[0] <= 0 or evals[-1]/evals[0] > 1e10:
        raise InsufficientSupportError('GMM moment covariance singular or ill-conditioned')
    gx = (z.T @ x) / n
    gy = (z.T @ y) / n
    # Solve linear systems instead of forming an explicit inverse.
    wg = np.linalg.solve(moment_cov, gx)
    wy = np.linalg.solve(moment_cov, gy)
    normal = gx.T @ wg
    e = np.linalg.eigvalsh(normal)
    if e[0] <= 0 or e[-1]/e[0] > 1e10:
        raise InsufficientSupportError('Two-step GMM structural moment system not identified')
    beta = np.linalg.solve(normal, gx.T @ wy)
    g = (z.T @ (y - x @ beta))/n
    stat = float(n * g @ np.linalg.solve(moment_cov, g))
    if not np.isfinite(stat) or stat < -1e-9:
        raise InsufficientSupportError('Invalid auxiliary GMM J statistic')
    stat = max(0.0, stat)
    return GMMOveridentificationTest(stat, float(stats.chi2.sf(stat, l-k)), l-k,
                                    tuple(map(float, beta)), method=method_label)


@dataclass(frozen=True)
class EffectiveFTest:
    """Montiel Olea & Pflueger (2013, JBES) heteroskedastic/cluster-robust Effective F."""

    f_effective: float
    k_effective: float
    n_instruments: int
    critical_values_tsls_bias: dict[str, float]  # tau = 5%, 10%, 20%, 30%
    covariance: str


def _partial_out(mat: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residualize columns of mat on exogenous controls (including intercept) via QR."""
    q_mat, _ = np.linalg.qr(controls, mode="reduced")
    return mat - q_mat @ (q_mat.T @ mat)


def effective_f_statistic(
    x_end: np.ndarray,
    z_excl: np.ndarray,
    w_controls: np.ndarray,
    *,
    kind: str = "HC0",
    groups: np.ndarray | None = None,
) -> EffectiveFTest:
    """Montiel Olea & Pflueger (2013) Effective F-statistic F_eff with critical values.

    Valid under heteroskedasticity and clustering for a single endogenous regressor.
    ``F_eff = (pi_hat' Q_ZZ pi_hat) / tr(W_2)`` where ``W_2`` is the robust/cluster
    asymptotic variance of orthonormalized first-stage scores.
    """
    x_e = np.asarray(x_end, dtype=float).reshape(-1)
    z_e = np.asarray(z_excl, dtype=float)
    if z_e.ndim == 1:
        z_e = z_e.reshape(-1, 1)
    w_c = np.asarray(w_controls, dtype=float)
    if w_c.ndim == 1:
        w_c = w_c.reshape(-1, 1)
    n, k = z_e.shape
    if len(x_e) != n or w_c.shape[0] != n:
        raise InvalidDesignError("Effective F inputs must have equal row counts")

    # Partial out exogenous controls W (including intercept) and orthonormalize Z_perp
    z_perp = _partial_out(z_e, w_c)
    x_perp = _partial_out(x_e[:, None], w_c).reshape(-1)
    # Orthonormal basis Q_Z such that (1/n) Z_tilde' Z_tilde = I_k
    q_z, r_z = np.linalg.qr(z_perp, mode="reduced")
    if np.linalg.matrix_rank(r_z) < k:
        raise NotIdentifiedError("Excluded instruments are collinear with exogenous controls")
    z_tilde = q_z * np.sqrt(n)  # (1/n) Z_tilde' Z_tilde = I_k

    # First-stage regression of x_perp on z_tilde
    pi_hat = (z_tilde.T @ x_perp) / n  # length k
    v_hat = x_perp - z_tilde @ pi_hat

    # Compute W_2 = (1/n) * robust/cluster meat of z_tilde * v_hat
    scores = z_tilde * v_hat[:, None]
    if groups is None:
        if kind not in ("HC0", "HC1"):
            raise InvalidDesignError("Unclustered effective F requires HC0 or HC1")
        w2 = (scores.T @ scores) / n
        if kind == "HC1":
            w2 *= n / max(1, n - k - w_c.shape[1])
    else:
        if kind not in ("CR0", "CR1"):
            raise InvalidDesignError("Clustered effective F requires CR0 or CR1")
        labels = np.asarray(groups)
        _, inv = np.unique(labels, return_inverse=True)
        g_cnt = int(inv.max() + 1)
        if g_cnt < 10:
            raise InsufficientSupportError("Clustered effective F requires at least 10 clusters")
        c_scores = np.zeros((g_cnt, k), dtype=float)
        np.add.at(c_scores, inv, scores)
        w2 = (c_scores.T @ c_scores) / n
        if kind == "CR1":
            w2 *= (g_cnt / (g_cnt - 1.0)) * ((n - 1.0) / max(1, n - k - w_c.shape[1]))

    tr_w2 = float(np.trace(w2))
    if not np.isfinite(tr_w2) or tr_w2 <= 1e-16:
        raise InsufficientSupportError("Degenerate first-stage variance for Effective F")
    f_eff = float((n * float(pi_hat @ pi_hat)) / tr_w2)

    # Effective degrees of freedom K_eff(x) per Montiel Olea & Pflueger (2013, JBES) eq. (10),
    # evaluated at the conservative simplified worst-case Nagar/TSLS benchmark x = 1 / tau
    # (since worst-case B_TSLS(W) <= 1 for all W; exact MOP eq. (10) Patnaik approximation).
    tr_w2_sq = float(np.sum(w2 * w2))
    max_eig = float(np.max(np.linalg.eigvalsh(w2)))
    k_eff = float((tr_w2 * tr_w2) / max(tr_w2_sq, 1e-16))
    crit_vals: dict[str, float] = {}
    for pct, tau in (("5%", 0.05), ("10%", 0.10), ("20%", 0.20), ("30%", 0.30)):
        x_val = 1.0 / tau
        denom_x = tr_w2_sq + 2.0 * x_val * tr_w2 * max_eig
        k_eff_x = float((tr_w2 * tr_w2 * (1.0 + 2.0 * x_val)) / max(denom_x, 1e-16))
        k_eff_x = max(k_eff_x, 1.0)
        crit_vals[pct] = float(stats.ncx2.ppf(0.95, df=k_eff_x, nc=k_eff_x * x_val) / k_eff_x)

    return EffectiveFTest(
        f_effective=f_eff,
        k_effective=k_eff,
        n_instruments=k,
        critical_values_tsls_bias=crit_vals,
        covariance=kind,
    )


def tf_critical_value(first_stage_f: float, *, alpha: float = 0.05) -> float:
    """Lee, McCrary, Moreira & Porter (2022, AER) tF critical value for just-identified IV.

    Smooth monotone interpolation of Table 3A (alpha=0.05) and Table 3B (alpha=0.01)
    tF critical values as a function of the first-stage F-statistic. Uses smooth
    pole-product scaling near the asymptote (F -> chi2_{1,1-alpha}^+) so critical
    values diverge properly rather than clamping, and returns +inf for F <= chi2_{1,1-alpha}.
    """
    if isinstance(first_stage_f, bool) or not np.isfinite(first_stage_f) or first_stage_f < 0:
        raise InvalidDesignError("first_stage_f must be finite and nonnegative")
    if np.isclose(alpha, 0.05):
        # Lee et al. (2022) AER Table 3A knots: (F, c_0.05(F))
        pole = 3.8415
        knots_f = np.array([4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 104.7])
        knots_c = np.array([15.65, 5.56, 4.21, 3.43, 3.12, 2.75, 2.57, 2.38, 2.20, 1.96])
        if first_stage_f <= pole:
            return float("inf")
        if first_stage_f < knots_f[0]:
            return float(knots_c[0] * np.sqrt((knots_f[0] - pole) / (first_stage_f - pole)))
        if first_stage_f >= 104.7:
            return 1.96
        return float(np.interp(first_stage_f, knots_f, knots_c))
    if np.isclose(alpha, 0.01):
        pole = 6.6349
        knots_f = np.array([7.0, 10.0, 15.0, 25.0, 50.0, 100.0, 252.3])
        knots_c = np.array([14.90, 5.45, 4.18, 3.48, 3.04, 2.82, 2.576])
        if first_stage_f <= pole:
            return float("inf")
        if first_stage_f < knots_f[0]:
            return float(knots_c[0] * np.sqrt((knots_f[0] - pole) / (first_stage_f - pole)))
        if first_stage_f >= 252.3:
            return 2.576
        return float(np.interp(first_stage_f, knots_f, knots_c))
    raise InvalidDesignError("tF critical values are tabulated for alpha=0.05 and alpha=0.01")


@dataclass(frozen=True)
class LIMLResult:
    """LIML / Fuller k-class estimator result for a single endogenous regressor."""

    estimate: float
    se: float
    p_value: float
    ci: tuple[float, float]
    kappa: float
    fuller_alpha: float
    covariance: str


def liml_estimate(
    y: np.ndarray,
    x_end: np.ndarray,
    z_excl: np.ndarray,
    w_controls: np.ndarray,
    *,
    fuller_alpha: float = 0.0,
    alpha: float = 0.05,
    kind: str = "HC0",
    groups: np.ndarray | None = None,
) -> LIMLResult:
    """Exact Limited Information Maximum Likelihood (LIML) or Fuller(fuller_alpha) k-class IV.

    Solves the minimum generalized eigenvalue problem M_W Y_2 v = kappa M_Z Y_2 v
    for Y_2 = [y, x_end] after partialling out exogenous controls W, then sets
    kappa_eff = kappa_LIML - fuller_alpha / (n - L).
    """
    if isinstance(fuller_alpha, bool) or not np.isscalar(fuller_alpha) or not np.isfinite(fuller_alpha) or fuller_alpha < 0:
        raise InvalidDesignError("fuller_alpha must be finite and nonnegative (0 for LIML, 1 or 4 for Fuller)")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        raise InvalidDesignError("alpha must be finite and strictly between 0 and 1")
    if groups is None:
        if kind not in ("HC0", "HC1"):
            raise InvalidDesignError("Unclustered LIML/Fuller requires kind='HC0' or 'HC1'")
    else:
        if kind not in ("CR0", "CR1"):
            raise InvalidDesignError("Clustered LIML/Fuller requires kind='CR0' or 'CR1'")
    y_v = np.asarray(y, dtype=float).reshape(-1)
    x_v = np.asarray(x_end, dtype=float).reshape(-1)
    z_e = np.asarray(z_excl, dtype=float)
    if z_e.ndim == 1:
        z_e = z_e.reshape(-1, 1)
    w_c = np.asarray(w_controls, dtype=float)
    if w_c.ndim == 1:
        w_c = w_c.reshape(-1, 1)
    n = len(y_v)
    z_full = np.column_stack([w_c, z_e])
    l_total = z_full.shape[1]
    if n <= l_total + 1:
        raise InsufficientSupportError("Insufficient degrees of freedom for LIML/Fuller")

    y2 = np.column_stack([y_v, x_v])
    # Residualize Y2 on W alone (E_0 = M_W Y2) and on full Z = [W, Z_excl] (E_1 = M_Z Y2)
    e0 = _partial_out(y2, w_c)
    e1 = _partial_out(y2, z_full)
    w0 = e0.T @ e0  # 2 x 2
    w1 = e1.T @ e1  # 2 x 2
    evals_w1 = np.linalg.eigvalsh(w1)
    if evals_w1[0] <= 1e-14:
        raise InsufficientSupportError("LIML residual cross-product matrix is singular")
    # Generalized eigenvalues of w0 v = kappa w1 v via Cholesky of w1
    chol = np.linalg.cholesky(w1)
    chol_inv = np.linalg.solve(chol, np.eye(2))
    sym = chol_inv @ w0 @ chol_inv.T
    sym = 0.5 * (sym + sym.T)
    kappa_liml = float(np.min(np.linalg.eigvalsh(sym)))
    kappa_eff = kappa_liml - (float(fuller_alpha) / (n - l_total))

    # k-class estimator on partialled-out (x_perp, y_perp):
    # beta_k = (x' (I - kappa M_Z) y) / (x' (I - kappa M_Z) x) = (w0[0,1] - kappa * w1[0,1]) / (w0[1,1] - kappa * w1[1,1])
    denom = float(w0[1, 1] - kappa_eff * w1[1, 1])
    if abs(denom) <= 1e-14:
        raise InsufficientSupportError("LIML/Fuller denominator vanished (unidentified)")
    beta_end = float((w0[0, 1] - kappa_eff * w1[0, 1]) / denom)

    # Recover full structural residuals u = y - x_end * beta_end - W * gamma_W
    w_fit = least_squares(w_c, y_v - beta_end * x_v)
    u_struct = w_fit.residuals
    # Standard k-class robust sandwich (fixed-kappa asymptotic approximation; not Bekker many-instrument SE)
    x_k = e0[:, 1] - kappa_eff * e1[:, 1]
    scores = (x_k * u_struct) / denom
    if groups is None:
        var = float(scores @ scores)
        if kind == "HC1":
            var *= n / max(1, n - w_c.shape[1] - 1)
        se = float(np.sqrt(max(var, 0.0)))
        if not np.isfinite(se) or se <= 0.0:
            raise InsufficientSupportError("Degenerate LIML/Fuller standard error")
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        p_val = float(2.0 * stats.norm.sf(abs(beta_end / se)))
    else:
        inv, g_cnt = _validate_groups(groups, n)
        if g_cnt < 10:
            raise InsufficientSupportError("Clustered LIML requires at least 10 clusters")
        c_scores = np.bincount(inv, weights=scores, minlength=g_cnt)
        var = float(c_scores @ c_scores)
        if kind == "CR1":
            var *= (g_cnt / (g_cnt - 1.0)) * ((n - 1.0) / max(1, n - w_c.shape[1] - 1))
        se = float(np.sqrt(max(var, 0.0)))
        if not np.isfinite(se) or se <= 0.0:
            raise InsufficientSupportError("Degenerate clustered LIML/Fuller standard error")
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=g_cnt - 1))
        p_val = float(2.0 * stats.t.sf(abs(beta_end / se), df=g_cnt - 1))
    return LIMLResult(
        estimate=beta_end,
        se=se,
        p_value=p_val,
        ci=(beta_end - crit * se, beta_end + crit * se),
        kappa=kappa_eff,
        fuller_alpha=float(fuller_alpha),
        covariance=kind,
    )


@dataclass(frozen=True)
class CLRTestResult:
    """Moreira (2003) / Andrews-Moreira-Stock (2006) Conditional Likelihood Ratio test."""

    null_effect: float
    clr_statistic: float
    ar_statistic: float
    lm_statistic: float
    qt_conditioning: float
    p_value: float
    n_instruments: int


def moreira_clr_test(
    y: np.ndarray,
    x_end: np.ndarray,
    z_excl: np.ndarray,
    w_controls: np.ndarray,
    null_effect: float = 0.0,
    *,
    reps: int = 4000,
    seed: int = 42,
) -> CLRTestResult:
    """Moreira (2003) Conditional Likelihood Ratio (CLR) weak-IV-robust test.

    Computes exact S and T sufficient statistics for H0: beta = null_effect,
    the Moreira CLR statistic ``0.5 * (Q_S - Q_T + sqrt((Q_S + Q_T)^2 - 4*(Q_S*Q_T - Q_ST^2)))``,
    and its conditional p-value given Q_T via Monte Carlo draws from chi2(1) and chi2(k-1).
    """
    if isinstance(null_effect, bool) or not np.isscalar(null_effect) or not np.isfinite(null_effect):
        raise InvalidDesignError("null_effect must be a finite number")
    if type(reps) is not int or reps < 100:
        raise InvalidDesignError("reps must be an integer >= 100")
    if type(seed) is not int or seed < 0:
        raise InvalidDesignError("seed must be a nonnegative integer")
    y_v = np.asarray(y, dtype=float).reshape(-1)
    x_v = np.asarray(x_end, dtype=float).reshape(-1)
    z_e = np.asarray(z_excl, dtype=float)
    if z_e.ndim == 1:
        z_e = z_e.reshape(-1, 1)
    w_c = np.asarray(w_controls, dtype=float)
    if w_c.ndim == 1:
        w_c = w_c.reshape(-1, 1)
    n, k = z_e.shape
    z_full = np.column_stack([w_c, z_e])
    df_e = n - z_full.shape[1]
    if df_e <= 2:
        raise InsufficientSupportError("Insufficient degrees of freedom for CLR test")

    # Partial out controls W and orthonormalize excluded instruments Z_perp
    z_perp = _partial_out(z_e, w_c)
    q_z, r_z = np.linalg.qr(z_perp, mode="reduced")
    if np.linalg.matrix_rank(r_z) < k:
        raise NotIdentifiedError("Excluded instruments rank-deficient given controls")

    y2 = np.column_stack([y_v, x_v])
    e0 = _partial_out(y2, w_c)
    e1 = _partial_out(y2, z_full)
    omega = (e1.T @ e1) / df_e  # 2 x 2 reduced-form covariance estimate
    if np.linalg.det(omega) <= 1e-14:
        raise InsufficientSupportError("Reduced-form covariance matrix is singular")
    omega_inv = np.linalg.solve(omega, np.eye(2))

    b0 = np.array([1.0, -float(null_effect)], dtype=float)
    a0 = np.array([float(null_effect), 1.0], dtype=float)
    # Sufficient statistics S (k-vector) and T (k-vector) per Moreira (2003)
    zt_y2 = q_z.T @ e0  # k x 2
    s_scale = float(np.sqrt(b0 @ omega @ b0))
    t_scale = float(np.sqrt(a0 @ omega_inv @ a0))
    s_vec = (zt_y2 @ b0) / s_scale
    t_vec = (zt_y2 @ (omega_inv @ a0)) / t_scale

    q_s = float(s_vec @ s_vec)
    q_t = float(t_vec @ t_vec)
    q_st = float(s_vec @ t_vec)
    ar_stat = q_s
    lm_stat = (q_st * q_st / q_t) if q_t > 1e-14 else 0.0
    disc = max(0.0, (q_s + q_t) ** 2 - 4.0 * max(0.0, q_s * q_t - q_st * q_st))
    clr_stat = float(0.5 * (q_s - q_t + np.sqrt(disc)))

    # Conditional distribution under H0 given Q_T = q_t:
    # Let Z_1 ~ chi2(1), Z_{k-1} ~ chi2(k-1) independent; CLR* = 0.5*(Z1 + Z_{k-1} - q_t + sqrt((Z1 + Z_{k-1} + q_t)^2 - 4*Z_{k-1}*q_t))
    if k == 1:
        p_val = float(stats.chi2.sf(clr_stat, df=1))
    else:
        rng = np.random.default_rng(seed)
        z1 = rng.chisquare(df=1.0, size=reps)
        zk = rng.chisquare(df=float(k - 1), size=reps)
        tot = z1 + zk
        clr_draws = 0.5 * (tot - q_t + np.sqrt(np.maximum(0.0, (tot + q_t) ** 2 - 4.0 * zk * q_t)))
        p_val = float((1 + np.sum(clr_draws >= clr_stat)) / (reps + 1))

    return CLRTestResult(
        null_effect=float(null_effect),
        clr_statistic=clr_stat,
        ar_statistic=ar_stat,
        lm_statistic=lm_stat,
        qt_conditioning=q_t,
        p_value=p_val,
        n_instruments=k,
    )
