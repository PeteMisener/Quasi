"""Exploratory Monte Carlo calibration benchmark for Quasi's frontier econometric upgrades.

IMPORTANT: This script produces reproducible *exploratory simulation evidence* on
synthetic known-truth DGPs. It is NOT external R/Python reference parity and does
NOT constitute statistical certification.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np
import pandas as pd

import quasi
from quasi.diagnostics.density import cjm_density_test
from quasi.inference.covariance import cr2_covariance, ewc_har_covariance, hac_covariance
from quasi.inference.linear import least_squares
from quasi.inference.resampling import wild_cluster_bootstrap
from quasi.matching.psm import AIPWATT, PropensityScoreMatching
from quasi.synth.ascm import AugmentedSCM


def _coverage(hits: int, n: int) -> dict[str, float]:
    rate = hits / max(n, 1)
    mcse = math.sqrt(max(rate * (1.0 - rate), 0.0) / max(n, 1))
    return {"coverage": round(rate, 4), "mcse": round(mcse, 4), "reps": n}


def bench_its_serial(reps: int = 200, seed: int = 20261127) -> dict[str, Any]:
    """Compare NW-HAC vs EWC fixed-b HAR vs REML AR(1) under strong AR(1) & AR(2)."""
    rng = np.random.default_rng(seed)
    T = 96
    t = np.arange(T, dtype=float)
    step = (t >= 48).astype(float)
    post = np.maximum(0.0, t - 48.0)
    tau = 2.0

    results = {}
    for dgp_name, ar_coeffs in [("ar1_0.80", [0.80]), ("ar2_0.95_-0.25", [0.95, -0.25])]:
        hits = {"hac": 0, "hac_pw": 0, "ewc": 0, "ar1_mle": 0, "ar1_reml": 0}
        ok = {"hac": 0, "hac_pw": 0, "ewc": 0, "ar1_mle": 0, "ar1_reml": 0}
        failed = {"hac": 0, "hac_pw": 0, "ewc": 0, "ar1_mle": 0, "ar1_reml": 0}
        for _ in range(reps):
            e = np.zeros(T + 40)
            eps = rng.normal(0.0, 1.0, T + 40)
            for i in range(len(ar_coeffs), T + 40):
                e[i] = sum(c * e[i - k - 1] for k, c in enumerate(ar_coeffs)) + eps[i]
            y = 10.0 + 0.05 * t + tau * step + 0.0 * post + e[40:]
            df = pd.DataFrame({"t": np.arange(T), "y": y})
            for mode in ("hac", "hac_pw", "ewc", "ar1_mle", "ar1_reml"):
                try:
                    r = quasi.InterruptedTimeSeries(df, "y", "t", 48, error_model=mode).fit()
                    ok[mode] += 1
                    if r.ci_lower <= tau <= r.ci_upper:
                        hits[mode] += 1
                except Exception:
                    failed[mode] += 1
        results[dgp_name] = {
            m: {**_coverage(hits[m], ok[m]), "failed": failed[m], "attempted": reps}
            for m in hits
        }
    return results


def bench_did_dr_improved(reps: int = 120, seed: int = 20261128) -> dict[str, Any]:
    """Compare unadjusted vs dr_improved under covariate-dependent trend DGP."""
    rng = np.random.default_rng(seed)
    n_units = 160
    tau = 1.5
    hits = {"unadjusted": 0, "dr_improved": 0}
    ok = {"unadjusted": 0, "dr_improved": 0}
    failed = {"unadjusted": 0, "dr_improved": 0}
    biases = {"unadjusted": [], "dr_improved": []}
    for _ in range(reps):
        x = rng.uniform(-1.5, 1.5, n_units)
        p = 1.0 / (1.0 + np.exp(-0.9 * x))
        treated = rng.binomial(1, p)
        g = np.where(treated == 1, 3, 0)
        rows = []
        for u in range(n_units):
            fe = rng.normal(0.0, 0.5)
            for t in (1, 2, 3, 4):
                trend = 0.6 * (t - 1) * x[u]
                y = fe + trend + (tau if (g[u] == 3 and t >= 3) else 0.0) + rng.normal(0.0, 0.8)
                rows.append((u, t, g[u], x[u], y))
        df = pd.DataFrame(rows, columns=["u", "t", "g", "x", "y"])
        for mode, covs in [("unadjusted", None), ("dr_improved", ["x"])]:
            try:
                r = quasi.StaggeredDiD(
                    df, "y", "u", "t", first_treated="g", covariates=covs, est_method=mode, b_reps=99, seed=1
                ).fit()
                ok[mode] += 1
                biases[mode].append(r.estimate - tau)
                if r.ci_lower <= tau <= r.ci_upper:
                    hits[mode] += 1
            except Exception:
                failed[mode] += 1
    return {
        m: {
            **_coverage(hits[m], ok[m]),
            "mean_bias": round(float(np.mean(biases[m])), 4) if biases[m] else None,
            "failed": failed[m],
            "attempted": reps,
        }
        for m in hits
    }


def bench_rdd_mserd_rbc(reps: int = 120, seed: int = 20261129) -> dict[str, Any]:
    """Compare conventional vs native mserd + RBC + NN VCE on nonlinear curved RD DGP."""
    rng = np.random.default_rng(seed)
    n = 600
    tau = 2.0
    hits = {"conv_mserd": 0, "rbc_nn_mserd": 0}
    cjm_rejects = 0
    for _ in range(reps):
        x = rng.uniform(-1.0, 1.0, n)
        mu = 0.5 * x + 2.5 * x**2 - 1.2 * x**3 + tau * (x >= 0.0)
        y = mu + rng.normal(0.0, 0.5, n)
        df = pd.DataFrame({"x": x, "y": y})
        rc = quasi.SharpRDD(df, "y", "x", 0.0, bandwidth="mserd", inference="conventional").fit()
        rr = quasi.SharpRDD(df, "y", "x", 0.0, bandwidth="mserd", inference="bias_corrected", vce="nn").fit()
        if rc.ci_lower <= tau <= rc.ci_upper:
            hits["conv_mserd"] += 1
        if rr.ci_lower <= tau <= rr.ci_upper:
            hits["rbc_nn_mserd"] += 1
        cjm = cjm_density_test(x, cutoff=0.0)
        if cjm.p_value is not None and cjm.p_value < 0.05:
            cjm_rejects += 1
    return {
        "conv_mserd": _coverage(hits["conv_mserd"], reps),
        "rbc_nn_mserd": _coverage(hits["rbc_nn_mserd"], reps),
        "cjm_null_size": _coverage(cjm_rejects, reps),
    }


def bench_iv_weak_and_cr2(reps: int = 150, seed: int = 20261130) -> dict[str, Any]:
    """Compare Wald vs Moreira CLR under weak instruments + CR2 vs WCB under G=12 clusters."""
    rng = np.random.default_rng(seed)
    tau = 1.0
    hits_wald = 0
    hits_clr = 0
    n = 300
    for _ in range(reps):
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        v = 0.8 * u + math.sqrt(1.0 - 0.64) * rng.normal(0, 1, n)
        d = 0.14 * z1 + 0.14 * z2 + v  # weak instruments (F ~ 6)
        y = tau * d + u
        df = pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2})
        r = quasi.TwoStageLS(df, "y", "d", ["z1", "z2"]).fit()
        if r.ci_lower <= tau <= r.ci_upper:
            hits_wald += 1
        if r.clr_test(null_effect=tau, reps=1000, seed=42).p_value >= 0.05:
            hits_clr += 1

    # Few-cluster G=12 CR2 Satterthwaite & WCB coverage
    G, m = 12, 15
    hits_cr2 = 0
    hits_wcb = 0
    for r_idx in range(reps):
        g = np.repeat(np.arange(G), m)
        trt = np.repeat(np.array([0] * 6 + [1] * 6, dtype=float), m)
        X = np.column_stack([np.ones(G * m), trt])
        u_g = rng.normal(0.0, 1.0, G)[g]
        y = 1.0 + 0.0 * trt + u_g + rng.normal(0.0, 0.5, G * m)
        fit = least_squares(X, y)
        cr2 = cr2_covariance(X, fit.residuals, g)
        from scipy import stats
        crit = float(stats.t.ppf(0.975, cr2.satterthwaite_df[1]))
        se = math.sqrt(cr2.covariance[1, 1])
        if abs(fit.coefficients[1]) <= crit * se:
            hits_cr2 += 1
        wcb = wild_cluster_bootstrap(
            X, y, g, param_index=1, null_value=0.0, reps=199, weights="webb", seed=r_idx
        )
        if wcb.p_value >= 0.05:
            hits_wcb += 1

    return {
        "weak_iv_wald": _coverage(hits_wald, reps),
        "weak_iv_clr": _coverage(hits_clr, reps),
        "few_cluster_g12_cr2_satterthwaite": _coverage(hits_cr2, reps),
        "few_cluster_g12_wcb_webb": _coverage(hits_wcb, reps),
    }


def bench_psm_ai_and_ascm(reps: int = 100, seed: int = 20261131) -> dict[str, Any]:
    """Check Abadie-Imbens PSM SE coverage & Augmented SCM conformal interval coverage."""
    rng = np.random.default_rng(seed)
    tau = 2.0
    hits_ai = 0
    hits_aipw = 0
    for _ in range(reps):
        n = 350
        x = rng.normal(0, 1, n)
        p = 1.0 / (1.0 + np.exp(-0.6 * x))
        d = rng.binomial(1, p)
        y = 1.0 + 1.5 * x + tau * d + rng.normal(0, 1.0, n)
        df = pd.DataFrame({"y": y, "d": d, "x": x})
        psm = PropensityScoreMatching(df, "y", "d", ["x"], n_neighbors=3).fit()
        ai = psm.abadie_imbens_se()
        if ai["ci_lower"] <= tau <= ai["ci_upper"]:
            hits_ai += 1
        aipw = AIPWATT(df, "y", "d", ["x"], stabilized=True).fit()
        if aipw.ci_lower <= tau <= aipw.ci_upper:
            hits_aipw += 1

    # Augmented SCM conformal interval under stationary noise (T_pre=24, T_post=1)
    hits_conf = 0
    for _ in range(reps):
        T_pre, T_post, N_ctrl = 24, 1, 10
        T = T_pre + T_post
        factors = rng.normal(0, 1.0, (T, 2))
        loadings = rng.normal(0, 1.0, (N_ctrl, 2))
        w_true = np.zeros(N_ctrl)
        w_true[:3] = [0.5, 0.3, 0.2]
        Y_ctrl = loadings @ factors.T + rng.normal(0, 0.5, (N_ctrl, T))
        Y_trt = w_true @ Y_ctrl + rng.normal(0, 0.5, T)
        Y_trt[T_pre:] += tau
        rows = []
        for t in range(T):
            rows.append(("trt", t, float(Y_trt[t])))
            for j in range(N_ctrl):
                rows.append((f"d{j}", t, float(Y_ctrl[j, t])))
        df_s = pd.DataFrame(rows, columns=["unit", "time", "y"])
        res = AugmentedSCM(df_s, "y", "unit", "time", "trt", T_pre).fit()
        ci = res.conformal_interval(alpha=0.10)
        if ci.ci_lower <= tau <= ci.ci_upper:
            hits_conf += 1

    return {
        "psm_abadie_imbens_95": _coverage(hits_ai, reps),
        "aipw_stabilized_95": _coverage(hits_aipw, reps),
        "ascm_conformal_90": _coverage(hits_conf, reps),
    }


def main() -> None:
    report = {
        "status": "EXPLORATORY_SIMULATION_ONLY_NOT_CERTIFICATION",
        "its_serial_95": bench_its_serial(),
        "did_dr_improved_95": bench_did_dr_improved(),
        "rdd_mserd_rbc_95": bench_rdd_mserd_rbc(),
        "iv_weak_and_cr2_95": bench_iv_weak_and_cr2(),
        "psm_ai_and_ascm": bench_psm_ai_and_ascm(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
