"""Publication-ready dynamic event-study plotting."""

from __future__ import annotations

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats


def render_event_study_figure(
    periods: List[int],
    estimates: List[float],
    se: List[float],
    ci_simultaneous: Optional[List[Tuple[float, float]]] = None,
    period_labels: Optional[List[str]] = None,
    alpha: float = 0.05,
    reference_period: int = -1,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generates clean event-study plot with pointwise and simultaneous confidence bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    else:
        fig = ax.get_figure()

    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    conf_pct = 100.0 * (1.0 - alpha)
    conf_str = f"{int(conf_pct)}%" if float(conf_pct).is_integer() else f"{conf_pct:g}%"

    ax.axhline(0, color="#7F8C8D", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.axvline(reference_period + 0.5, color="#C0392B", linestyle="--", linewidth=1.2, alpha=0.9, label="Intervention / Anticipation Onset")

    pre_mask = [p <= reference_period for p in periods]
    post_mask = [p > reference_period for p in periods]

    # Pre-treatment leads (Slate)
    if any(pre_mask):
        ax.errorbar(
            [periods[i] for i, m in enumerate(pre_mask) if m],
            [estimates[i] for i, m in enumerate(pre_mask) if m],
            yerr=[crit * se[i] for i, m in enumerate(pre_mask) if m],
            fmt="o",
            color="#2C3E50",
            ecolor="#7F8C8D",
            elinewidth=1.2,
            capsize=2.5,
            label="Pre-Trend",
        )

    # Post-treatment dynamic effects (Teal)
    if any(post_mask):
        ax.errorbar(
            [periods[i] for i, m in enumerate(post_mask) if m],
            [estimates[i] for i, m in enumerate(post_mask) if m],
            yerr=[crit * se[i] for i, m in enumerate(post_mask) if m],
            fmt="o",
            color="#16A085",
            ecolor="#16A085",
            elinewidth=1.5,
            capsize=3.0,
            label="Dynamic ATT",
        )

    # Simultaneous confidence envelope
    if ci_simultaneous is not None:
        lowers = [ci[0] for ci in ci_simultaneous]
        uppers = [ci[1] for ci in ci_simultaneous]
        ax.fill_between(
            periods, lowers, uppers, color="#16A085", alpha=0.15, label=f"Simultaneous {conf_str} CI"
        )

    if period_labels is not None and len(period_labels) == len(periods):
        ax.set_xticks(periods)
        ax.set_xticklabels(period_labels, rotation=45, ha="right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")

    ax.set_xlabel("Event Time Relative to Treatment", fontsize=10, fontweight="medium")
    ax.set_ylabel("Treatment Effect (ATT)", fontsize=10, fontweight="medium")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    return fig, ax