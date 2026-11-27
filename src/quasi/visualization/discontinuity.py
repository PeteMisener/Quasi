"""Publication-ready Regression Discontinuity visualization."""

from __future__ import annotations

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from quasi.rdd.kernels import KERNEL_REGISTRY


def render_discontinuity_figure(
    X: np.ndarray,
    Y: np.ndarray,
    cutoff: float,
    bandwidth: float,
    estimate: float,
    se: float,
    p_value: float,
    kernel: str = "triangular",
    order: int = 1,
    n_bins: int = 30,
    is_fuzzy: bool = False,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders binned scatter points, O(N) local polynomial curves, and pointwise bands.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=100)
    else:
        fig = ax.get_figure()

    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    # Filter data within visualization window (1.5 * bandwidth)
    vis_window = 1.5 * bandwidth
    vis_mask = np.abs(X - cutoff) <= vis_window
    x_vis = X[vis_mask]
    y_vis = Y[vis_mask]

    # 1. Binned Scatter (Left and Right separately)
    def compute_binned_means(x_vals: np.ndarray, y_vals: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(x_vals) == 0:
            return np.array([]), np.array([])
        bins = max(1, bins)
        bin_edges = np.linspace(np.min(x_vals), np.max(x_vals), bins + 1)
        bin_ids = np.digitize(x_vals, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, bins - 1)
        mids = []
        means = []
        for b_idx in range(bins):
            in_bin = bin_ids == b_idx
            if np.any(in_bin):
                mids.append(np.mean(x_vals[in_bin]))
                means.append(np.mean(y_vals[in_bin]))
        return np.array(mids), np.array(means)

    left_mask = x_vis < cutoff
    right_mask = x_vis >= cutoff

    mids_l, means_l = compute_binned_means(x_vis[left_mask], y_vis[left_mask], max(2, n_bins // 2))
    mids_r, means_r = compute_binned_means(x_vis[right_mask], y_vis[right_mask], max(2, n_bins // 2))

    # Plot binned points
    ax.scatter(mids_l, means_l, color="#566573", alpha=0.75, s=28, edgecolors="none", zorder=3)
    ax.scatter(mids_r, means_r, color="#1B4F72", alpha=0.75, s=28, edgecolors="none", zorder=3)

    # 2. Local Polynomial Fitted Curves within [cutoff - h, cutoff + h] using O(N) WLS
    k_func = KERNEL_REGISTRY.get(kernel.lower(), KERNEL_REGISTRY["triangular"])

    def fit_and_predict_poly(side: str, grid_points: int = 150) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if side == "left":
            in_bw = (X < cutoff) & (X >= cutoff - bandwidth)
            grid = np.linspace(cutoff - bandwidth, cutoff - 1e-6 * max(1.0, bandwidth), grid_points)
        else:
            in_bw = (X >= cutoff) & (X <= cutoff + bandwidth)
            grid = np.linspace(cutoff, cutoff + bandwidth, grid_points)

        x_fit = X[in_bw] - cutoff
        y_fit = Y[in_bw]
        w_fit = k_func(x_fit / bandwidth)
        active = w_fit > 0.0
        x_fit = x_fit[active]
        y_fit = y_fit[active]
        w_fit = w_fit[active]

        Z_fit = np.vander(x_fit, N=order + 1, increasing=True)
        Z_grid = np.vander(grid - cutoff, N=order + 1, increasing=True)
        sqrt_w = np.sqrt(w_fit)
        Zw = Z_fit * sqrt_w[:, None]
        yw = y_fit * sqrt_w

        beta, _, _, _ = np.linalg.lstsq(Zw, yw, rcond=None)
        y_pred = Z_grid @ beta

        # O(N) HC0 sandwich variance via QR/SVD without dense N x N matrices
        residuals = y_fit - Z_fit @ beta
        _, r_mat = np.linalg.qr(Zw, mode="reduced")
        try:
            inv_r = np.linalg.solve(r_mat, np.eye(r_mat.shape[0]))
            bread = inv_r @ inv_r.T
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(Zw.T @ Zw)
        scores = Z_fit * (w_fit * residuals)[:, None]
        meat = scores.T @ scores
        cov_beta = bread @ meat @ bread
        pred_var = np.einsum("ij,jk,ik->i", Z_grid, cov_beta, Z_grid)
        pred_se = np.sqrt(np.maximum(pred_var, 0.0))

        return grid, y_pred, pred_se

    grid_l, pred_l, se_l = fit_and_predict_poly("left")
    grid_r, pred_r, se_r = fit_and_predict_poly("right")

    # Plot lines and shaded confidence envelopes
    ax.plot(grid_l, pred_l, color="#2C3E50", linewidth=2.0, label="Local Fit (Left)", zorder=4)
    ax.fill_between(grid_l, pred_l - crit * se_l, pred_l + crit * se_l, color="#2C3E50", alpha=0.15, zorder=2)

    ax.plot(grid_r, pred_r, color="#16A085", linewidth=2.2, label="Local Fit (Right)", zorder=4)
    ax.fill_between(grid_r, pred_r - crit * se_r, pred_r + crit * se_r, color="#16A085", alpha=0.18, zorder=2)

    # 3. Cutoff Threshold Line & Effective Bandwidth Shading
    ax.axvline(cutoff, color="#C0392B", linestyle="--", linewidth=1.2, alpha=0.9, label=f"Cutoff (c = {cutoff:g})")
    ax.axvspan(cutoff - bandwidth, cutoff + bandwidth, color="#EAEDED", alpha=0.35, zorder=1, label="Bandwidth Window [c ± h]")

    # 4. Annotation Box
    sign = "+" if estimate >= 0 else ""
    est_label = "Fuzzy RD Wald" if is_fuzzy else "Sharp RD Jump"
    info_text = (
        f"{est_label}: {sign}{estimate:.4f}\n"
        f"Std. Err: {se:.4f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Bandwidth (h): {bandwidth:.3f}"
    )
    ax.text(
        0.03,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.9),
        zorder=5,
    )

    # Styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")

    ax.set_xlabel("Running Variable (X)", fontsize=10.5, fontweight="medium")
    ax.set_ylabel("Outcome (Y)", fontsize=10.5, fontweight="medium")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()

    return fig, ax
