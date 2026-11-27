"""Publication-ready Synthetic Control and Synthetic DiD visualization."""

from __future__ import annotations

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def render_synth_figure(
    time_labels: List[str],
    actual_trajectory: np.ndarray,
    synthetic_trajectory: np.ndarray,
    intervention_idx: int,
    placebo_gaps: Optional[np.ndarray] = None,
    ax_traj: Optional[plt.Axes] = None,
    ax_gap: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Renders synchronized dual-panel trajectory and treatment gap plots with placebo envelopes."""
    if ax_traj is not None and ax_gap is not None:
        ax1, ax2 = ax_traj, ax_gap
        fig = ax1.get_figure()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.0), dpi=100, sharex=True)

    x_indices = np.arange(len(time_labels))
    interv_label_x = x_indices[intervention_idx] - 0.5 if intervention_idx < len(x_indices) else x_indices[-1]

    # --- Panel 1: Trajectory Levels ---
    ax1.plot(x_indices, actual_trajectory, color="#1B4F72", linewidth=2.2, label="Treated Unit", zorder=3)
    ax1.plot(x_indices, synthetic_trajectory, color="#E74C3C", linewidth=2.0, linestyle="--", label="Synthetic Counterfactual", zorder=3)
    ax1.axvline(interv_label_x, color="#7F8C8D", linestyle=":", linewidth=1.2, label="Treatment Start")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")
    ax1.set_ylabel("Outcome Level (Y)", fontsize=10, fontweight="medium")
    ax1.set_title("Synthetic Control Trajectory & Treatment Gap", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(frameon=False, loc="upper left", fontsize=9)

    # --- Panel 2: Treatment Gap (Actual - Synthetic) ---
    gap = actual_trajectory - synthetic_trajectory
    
    if placebo_gaps is not None:
        p_arr = np.asarray(placebo_gaps, dtype=float)
        if p_arr.ndim == 2 and p_arr.shape[0] > 0 and p_arr.shape[1] == len(x_indices):
            # Compute in-space placebo envelope (95% range)
            p_lower = np.percentile(p_arr, 2.5, axis=0)
            p_upper = np.percentile(p_arr, 97.5, axis=0)
            ax2.fill_between(x_indices, p_lower, p_upper, color="#BDC3C7", alpha=0.3, label="Donor Placebo 95% Range", zorder=1)

    ax2.plot(x_indices, gap, color="#16A085", linewidth=2.2, label="Estimated Gap (ATT)", zorder=3)
    ax2.axhline(0, color="#7F8C8D", linestyle="-", linewidth=0.8, alpha=0.8)
    ax2.axvline(interv_label_x, color="#7F8C8D", linestyle=":", linewidth=1.2)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")
    ax2.set_ylabel("Treatment Effect (Gap)", fontsize=10, fontweight="medium")
    
    # Format x-ticks
    step = max(1, len(time_labels) // 10)
    ax2.set_xticks(x_indices[::step])
    ax2.set_xticklabels([time_labels[i] for i in x_indices[::step]], rotation=30, ha="right", fontsize=9)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig, (ax1, ax2)