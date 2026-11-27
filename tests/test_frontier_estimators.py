"""Comprehensive verification of Phase 3 frontier econometric upgrades across all 6 families."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import quasi
from quasi.did.conditional import improved_dr_panel
from quasi.diagnostics.density import cjm_density_test
from quasi.iv.inference import tf_critical_value
from quasi.rdd.bandwidth import select_cct_bandwidth, select_mse_bandwidth
from quasi.rdd.local_inference import rd_inference


# --- 1. DiD: Improved DR (Sant'Anna & Zhao 2020), Unconditional Cohort IF, HonestDiD ---

def test_improved_dr_panel_exact_covariate_balance_and_double_robustness():
    rng = np.random.default_rng(301)
    n = 600
    x = rng.normal(size=(n, 2))
    ps_true = 1.0 / (1.0 + np.exp(-(0.3 + 0.5 * x[:, 0] - 0.4 * x[:, 1])))
    d = rng.binomial(1, ps_true)
    # Nonlinear outcome (outcome linear regression misspecified, propensity correctly specified)
    dy = 1.0 + 0.8 * x[:, 0] + 0.4 * (x[:, 0] ** 2) + 2.5 * d + rng.normal(scale=0.6, size=n)

    cell = improved_dr_panel(dy, d, x)
    assert abs(cell.att - 2.5) < 0.25
    assert np.isfinite(cell.influence).all()
    assert abs(float(np.mean(cell.influence))) < 1e-8


def test_staggered_did_improved_dr_unconditional_se_and_honest_did():
    rng = np.random.default_rng(302)
    rows = []
    # Two cohorts with strong treatment effect heterogeneity (+1.0 vs +5.0)
    for u in range(120):
        x1 = rng.normal()
        cohort = 3 if u < 40 else (4 if u < 80 else None)
        eff = 1.0 if cohort == 3 else (5.0 if cohort == 4 else 0.0)
        fe = rng.normal()
        for t in range(6):
            treated = int(cohort is not None and t >= cohort)
            y = 2.0 + fe + 0.3 * t + 0.5 * x1 * t + eff * treated + rng.normal(scale=0.4)
            rows.append({"unit": u, "time": t, "treated": treated, "y": y, "x1": x1})
    df = pd.DataFrame(rows)

    res_imp = quasi.estimate(
        df,
        design=quasi.PanelDesign(
            outcome="y", unit="unit", time="time", treatment="treated",
            covariates=("x1",), est_method="dr_improved", b_reps=49,
        ),
    )
    assert res_imp.est_method == "dr_improved"
    assert res_imp.unconditional_se is not None
    # Under strong cohort-effect heterogeneity, unconditional SE >= sample-conditional SE
    assert res_imp.unconditional_se >= res_imp.se
    assert res_imp.event_study_cov is not None

    # Rambachan & Roth (2023) HonestDiD sensitivity
    sens_rm = res_imp.sensitivity_analysis(method="relative_magnitudes", m_grid=[0.0, 0.5, 1.0, 2.0])
    sens_sd = res_imp.sensitivity_analysis(method="smoothness", m_grid=[0.0, 0.2, 0.5, 1.5])
    assert len(sens_rm.table) == 4
    assert len(sens_sd.table) == 4
    # Robust CI width must increase monotonically with M
    widths_sd = sens_sd.table["robust_ci_upper"] - sens_sd.table["robust_ci_lower"]
    assert np.all(np.diff(widths_sd) >= -1e-9)


# --- 2. RDD: Native Scale-Invariant CCT Bandwidths, b < h Fix, NN VCE, CJM Density, Fuzzy AR ---

def test_rdd_native_mserd_cerrd_and_scale_invariance():
    rng = np.random.default_rng(303)
    x = rng.uniform(-2.0, 2.0, 1200)
    y = 2.0 + 1.5 * (x >= 0) + 0.5 * x + 0.4 * (x**2) + rng.normal(scale=0.5, size=len(x))

    sel_mse = select_mse_bandwidth(x, y, 0.0, selector="mserd")
    sel_cer = select_mse_bandwidth(x, y, 0.0, selector="cerrd")
    assert 0.0 < sel_cer.h < sel_mse.h  # CER bandwidth is strictly smaller than MSE

    # Exact scale invariance: X -> 50*X, Y -> 20*Y => h_scaled == 50 * h_orig
    sel_scaled = select_mse_bandwidth(50.0 * x, 20.0 * y, 0.0, selector="mserd")
    assert sel_scaled.h == pytest.approx(50.0 * sel_mse.h, rel=1e-5)

    # Legacy select_cct_bandwidth is now also strictly scale invariant
    h1, _, _ = select_cct_bandwidth(x, y, 0.0)
    h2, _, _ = select_cct_bandwidth(50.0 * x, 20.0 * y, 0.0)
    assert h2 == pytest.approx(50.0 * h1, rel=1e-6)

    df = pd.DataFrame({"x": x, "y": y})
    res = quasi.estimate(df, design=quasi.RDDDesign(outcome="y", running_var="x", cutoff=0.0, bandwidth="mserd"))
    assert res.inference_details["bandwidth_selection"] == "native_mserd"
    assert abs(res.estimate - 1.5) < 0.25


def test_rdd_bias_bandwidth_smaller_than_h_and_nn_vce_and_fuzzy_ar():
    rng = np.random.default_rng(304)
    n = 1000
    x = rng.uniform(-2.0, 2.0, n)
    d = (rng.uniform(size=n) < np.where(x >= 0, 0.8, 0.2)).astype(float)
    y = 1.0 + 2.0 * d + 0.5 * x + rng.normal(scale=0.5, size=n)

    # b < h must NOT zero out residuals in b < |x| <= h
    fit_b_small = rd_inference(x, y, h=1.2, kernel="triangular", degree=1, bias_bandwidth=0.8)
    assert np.isfinite(fit_b_small["se"]) and fit_b_small["se"] > 0.0

    # Nearest-neighbor VCE (CCT 2014) and HC3
    fit_nn = rd_inference(x, y, h=1.0, kernel="triangular", degree=1, vce="nn")
    fit_hc3 = rd_inference(x, y, h=1.0, kernel="triangular", degree=1, vce="hc3")
    assert fit_nn["inference_method"] == "conventional_nn"
    assert fit_hc3["se"] > 0 and fit_nn["se"] > 0

    df = pd.DataFrame({"x": x, "y": y, "d": d})
    fuzzy = quasi.FuzzyRDD(df, "y", "x", 0.0, treatment="d", bandwidth=1.0, vce="nn").fit()
    ar_true = fuzzy.ar_test(null_effect=2.0)
    ar_zero = fuzzy.ar_test(null_effect=0.0)
    assert ar_true["p_value"] > 0.05
    assert ar_zero["p_value"] < 0.01


def test_cjm_density_test_distinguishes_smooth_vs_manipulated():
    rng = np.random.default_rng(305)
    x_clean = rng.uniform(-2.0, 2.0, 2500)
    res_clean = cjm_density_test(x_clean, cutoff=0.0)
    assert res_clean.p_value is not None and res_clean.p_value > 0.05

    # Strong manipulation: 2x density immediately right of cutoff
    x_left = rng.uniform(-2.0, 0.0, 1000)
    x_right = rng.uniform(0.0, 2.0, 2000)
    res_manip = cjm_density_test(np.concatenate([x_left, x_right]), cutoff=0.0)
    assert res_manip.p_value is not None and res_manip.p_value < 0.01


# --- 3. IV: Effective F, tF, LIML/Fuller, Moreira CLR, Clustered Hansen J ---

def test_iv_effective_f_tf_liml_clr_and_clustered_j():
    rng = np.random.default_rng(306)
    n = 500
    groups = np.arange(n) % 25
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.6 * z1 + 0.5 * z2 + 0.3 * w + 0.7 * u + rng.normal(scale=0.6, size=n)
    y = 1.0 + 2.0 * x + 0.4 * w + 0.7 * u + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "w": w, "g": groups})

    # Overidentified IV
    res_over = quasi.TwoStageLS(df, "y", "x", ["z1", "z2"], covariates=["w"], cluster="g").fit()
    eff = res_over.effective_f()
    assert eff.f_effective > 20.0
    assert eff.critical_values_tsls_bias["5%"] > eff.critical_values_tsls_bias["10%"]

    liml = res_over.liml(fuller_alpha=0.0)
    fuller = res_over.liml(fuller_alpha=1.0)
    assert abs(liml.estimate - 2.0) < 0.25
    assert abs(fuller.estimate - 2.0) < 0.25
    assert liml.kappa >= 1.0 - 1e-8

    res_over_uncl = quasi.TwoStageLS(df, "y", "x", ["z1", "z2"], covariates=["w"]).fit()
    clr_true = res_over_uncl.clr_test(null_effect=2.0, reps=1000)
    clr_false = res_over_uncl.clr_test(null_effect=0.0, reps=1000)
    assert clr_true.p_value > 0.05
    assert clr_false.p_value < 0.01

    j_cl = res_over.gmm_overidentification_test(allow_clustered=True)
    assert j_cl.degrees_of_freedom == 1
    assert j_cl.p_value > 0.01

    # Just-identified IV: LIML == 2SLS exactly, and tF CI widens properly
    res_just = quasi.TwoStageLS(df, "y", "x", ["z1"], covariates=["w"]).fit()
    liml_just = res_just.liml(fuller_alpha=0.0)
    assert liml_just.estimate == pytest.approx(res_just.estimate, abs=1e-9)
    tf_lo, tf_hi = res_just.tf_confidence_interval()
    assert tf_lo <= res_just.ci_lower + 1e-9 and tf_hi >= res_just.ci_upper - 1e-9
    assert np.isinf(tf_critical_value(3.0, alpha=0.05))


# --- 4. ITS: EWC HAR, Prewhitened HAC, REML AR1, In-Time Placebo, ControlledITS ---

def test_its_ewc_reml_placebo_and_controlled_its():
    rng = np.random.default_rng(307)
    n = 60
    t = np.arange(n)
    y = 10.0 + 0.2 * t + 3.0 * (t >= 30) + 0.1 * np.maximum(0, t - 30) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"t": t, "y": y})

    for model in ("ewc", "hac_pw", "ar1_reml"):
        res = quasi.estimate(df, design=quasi.ITSDesign(outcome="y", time="t", intervention_date=30, error_model=model))
        assert abs(res.level_change - 3.0) < 0.8
        assert np.isfinite(res.level_se) and res.level_se > 0

    ewc_res = quasi.InterruptedTimeSeries(df, "y", "t", 30, error_model="ewc").fit()
    placebos = ewc_res.in_time_placebo()
    assert len(placebos) >= 2
    assert set(["placebo_date", "level_change", "level_p"]) <= set(placebos.columns)

    # Controlled ITS (CITS)
    y_ctrl = 8.0 + 0.2 * t + 1.0 * (t >= 30) + rng.normal(scale=0.4, size=n)  # common shock +1
    y_trt = 10.0 + 0.2 * t + 4.0 * (t >= 30) + rng.normal(scale=0.4, size=n)  # common +1 + true +3
    cits_df = pd.concat([
        pd.DataFrame({"t": t, "g": 1, "y": y_trt}),
        pd.DataFrame({"t": t, "g": 0, "y": y_ctrl}),
    ], ignore_index=True)
    cits = quasi.ControlledITS(cits_df, outcome="y", time="t", group="g", intervention_date=30).fit()
    assert abs(cits.level_diff - 3.0) < 0.7
    assert cits.pre_trend_p > 0.05


# --- 5. Synthetic Control, Augmented SCM, SDiD Sparsify & Conformal Intervals ---

def test_scm_ascm_sdid_conformal_and_sparsify():
    rng = np.random.default_rng(308)
    donors = [f"d{i}" for i in range(10)]
    t_len = 25
    split = 20
    factors = rng.normal(size=(2, t_len))
    rows = []
    w_true = np.array([0.5, 0.3, 0.2] + [0.0] * 7)
    donor_y = np.zeros((10, t_len))
    for i in range(10):
        load = rng.uniform(0.5, 1.5, size=2)
        donor_y[i] = 5.0 + load @ factors + rng.normal(scale=0.15, size=t_len)
        for t in range(t_len):
            rows.append({"unit": donors[i], "time": t, "y": donor_y[i, t]})
    # Treated unit has level offset + true effect 4.0 post-split
    y_trt = 1.5 + w_true @ donor_y + np.where(np.arange(t_len) >= split, 4.0, 0.0) + rng.normal(scale=0.15, size=t_len)
    for t in range(t_len):
        rows.append({"unit": "treated", "time": t, "y": y_trt[t]})
    df = pd.DataFrame(rows)

    scm = quasi.SyntheticControl(df, "y", "unit", "time", "treated", split, n_placebos=4).fit()
    itp = scm.in_time_placebo()
    assert np.isfinite(itp["placebo_post_average_gap"])
    conf_scm = scm.conformal_interval(alpha=0.10)
    assert conf_scm.ci_lower <= conf_scm.ci_upper

    ascm = quasi.AugmentedSCM(df, "y", "unit", "time", "treated", split).fit()
    assert ascm.pre_rmspe <= scm.pre_rmspe + 1e-6  # ridge debiasing improves pre-treatment fit
    assert abs(ascm.estimate - 4.0) < 0.8
    conf_ascm = ascm.conformal_interval(alpha=0.10)
    assert conf_ascm.ci_lower <= conf_ascm.ci_upper

    sdid_sp = quasi.SyntheticDiD(df, "y", "unit", "time", "treated", split, n_placebos=3, sparsify="default").fit()
    assert sdid_sp.weight_details["sparsification"] == "default"
    assert abs(sdid_sp.estimate - 4.0) < 0.6
    conf_sdid = sdid_sp.conformal_interval(alpha=0.10)
    assert conf_sdid.ci_lower <= 4.0 <= conf_sdid.ci_upper


# --- 6. Observational: Abadie-Imbens SE, OVB Sensitivity, Stabilized & ML AIPW ---

def test_observational_abadie_imbens_se_ovb_sensitivity_and_ml_aipw():
    rng = np.random.default_rng(309)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.2 + 0.6 * x1 - 0.5 * x2))))
    y = 1.0 + 0.8 * x1 - 0.4 * x2 + 2.5 * d + rng.normal(scale=0.8, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})

    psm = quasi.PropensityScoreMatching(df, "y", "d", ["x1", "x2"], n_neighbors=2).fit()
    ai = psm.abadie_imbens_se(l_matches=2)
    assert np.isfinite(ai["se"]) and ai["se"] > 0
    assert ai["ci_lower"] < 2.5 < ai["ci_upper"]

    aipw = quasi.AIPWATT(df, "y", "d", ["x1", "x2"]).fit()
    ovb = aipw.ovb_sensitivity(q=1.0, alpha=0.05)
    assert 0.0 < ovb.rv_q_alpha < ovb.rv_q < 1.0
    # At r2dz = r2yz = RV_q, Cinelli & Hazlett Theorem 2 implies adjusted estimate hits ~0
    est_zero, _, _ = ovb.adjusted_estimate(ovb.rv_q, ovb.rv_q, direction=-1)
    assert abs(est_zero) < 1e-6

    # Stabilized cross-fitted AIPW with a simple custom learner adhering to fit/predict_proba/predict protocol
    class RidgeMeanLearner:
        def fit(self, X, y_v):
            z = np.column_stack([np.ones(len(X)), X])
            self.coef_ = np.linalg.solve(z.T @ z + 0.1 * np.eye(z.shape[1]), z.T @ y_v)
            return self
        def predict(self, X):
            z = np.column_stack([np.ones(len(X)), X])
            return z @ self.coef_

    aipw_ml = quasi.AIPWATT(
        df, "y", "d", ["x1", "x2"], n_splits=3, fold_seed=11,
        stabilized=True, outcome_learner=RidgeMeanLearner(),
    ).fit()
    assert aipw_ml.model_meta["ml_nuisance_learners"] is True
    assert aipw_ml.model_meta["stabilized_hajek_weights"] is True
    assert abs(aipw_ml.estimate - 2.5) < 0.35
