"""Publication-ready Love Plot for covariate balance assessment."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def render_love_figure(
    smd_unadjusted: Dict[str, float],
    smd_adjusted: Dict[str, float],
    threshold: float = 0.10,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders Standardized Mean Difference (SMD) Love plot before and after matching.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, max(4.0, len(smd_unadjusted) * 0.45)), dpi=100)
    else:
        fig = ax.get_figure()

    covariates = list(smd_unadjusted.keys())
    y_pos = np.arange(len(covariates))

    unadj_vals = [smd_unadjusted[c] for c in covariates]
    adj_vals = [smd_adjusted[c] for c in covariates]

    # Connecting horizontal guide line between unadjusted and adjusted points
    for i, y in enumerate(y_pos):
        ax.plot([unadj_vals[i], adj_vals[i]], [y, y], color="#BDC3C7", linewidth=1.2, zorder=1)

    # Plot Unadjusted (hollow slate circle)
    ax.scatter(unadj_vals, y_pos, facecolors="none", edgecolors="#566573", s=50, linewidth=1.5, label="Unadjusted", zorder=3)

    # Plot Adjusted (solid teal circle)
    ax.scatter(adj_vals, y_pos, color="#16A085", s=50, label="Matched / Reweighted", zorder=4)

    # Balance threshold vertical line (typically SMD = 0.10)
    ax.axvline(threshold, color="#E74C3C", linestyle="--", linewidth=1.0, label=f"Threshold (SMD = {threshold:g})")
    ax.axvline(0, color="#7F8C8D", linestyle="-", linewidth=0.8, alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates, fontsize=9.5)
    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.2, color="gray")

    ax.set_xlabel("Absolute Standardized Mean Difference (SMD)", fontsize=10, fontweight="medium")
    ax.set_title("Covariate Balance (Love Plot)", fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()

    return fig, ax