"""Publication-ready Interrupted Time Series visualization."""

from __future__ import annotations

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def render_its_figure(
    time_labels: List[str],
    y_actual: np.ndarray,
    y_fitted: np.ndarray,
    y_counterfactual: np.ndarray,
    intervention_idx: int,
    outcome_name: str,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders actual time series, factual segmented fit, and counterfactual projection.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=100)
    else:
        fig = ax.get_figure()

    T = len(y_actual)
    t_idx = np.arange(T)

    # 1. Observed Data Points
    ax.scatter(t_idx, y_actual, color="#7F8C8D", alpha=0.65, s=24, edgecolors="none", label="Observed Outcome")

    # 2. Pre-intervention Fit
    t_pre = t_idx[:intervention_idx]
    ax.plot(t_pre, y_fitted[:intervention_idx], color="#1B4F72", linewidth=2.2, label="Fitted Pre-Trend")

    # 3. Post-intervention Fit
    t_post = t_idx[intervention_idx:]
    ax.plot(t_post, y_fitted[intervention_idx:], color="#16A085", linewidth=2.4, label="Fitted Post-Trend")

    # 4. Counterfactual Projection (Dashed Red/Slate)
    ax.plot(
        t_post,
        y_counterfactual[intervention_idx:],
        color="#E74C3C",
        linestyle="--",
        linewidth=2.0,
        label="Projected Counterfactual",
    )

    # 5. Intervention Boundary Line
    interv_boundary = intervention_idx - 0.5
    ax.axvline(interv_boundary, color="#C0392B", linestyle=":", linewidth=1.4, label="Intervention")

    # Styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")

    # Format X-ticks
    step = max(1, T // 10)
    ax.set_xticks(t_idx[::step])
    ax.set_xticklabels([time_labels[i] for i in t_idx[::step]], rotation=30, ha="right", fontsize=9)

    ax.set_xlabel("Time Period", fontsize=10, fontweight="medium")
    ax.set_ylabel(outcome_name, fontsize=10, fontweight="medium")
    ax.set_title("Interrupted Time Series: Actual vs. Counterfactual Trajectory", fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="best", fontsize=8.5)
    fig.tight_layout()

    return fig, ax