"""Base data structures and visualization routines for statistical power analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple
import numpy as np


@dataclass
class PowerResult:
    """
    Standardized result container for quasi-experimental power calculations.
    """
    design: str
    mde: float
    mde_standardized: float
    target_power: float
    alpha: float
    parameters: Dict[str, Any]
    curve_func_effect: Callable[[np.ndarray], np.ndarray]
    curve_func_sample: Callable[[np.ndarray], np.ndarray]
    sample_param_name: str
    sample_param_val: float

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"Ex-Ante Power Analysis: {self.design}",
            "=" * 70,
            f"Target Statistical Power: {self.target_power * 100:.1f}% (α = {self.alpha:.3f})",
            f"Minimum Detectable Effect (MDE): {self.mde:+.4f}",
            f"Standardized MDE (Cohen's d):    {self.mde_standardized:.4f} σ",
            "-" * 70,
            "Design Specifications & Assumptions:",
        ]
        for k, v in self.parameters.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"  • {k:<30}: {val_str}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def print_terminal(self) -> None:
        """Renders rich ANSI table to console."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"quasi.power: {self.design}")
        table.add_column("Metric / Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        table.add_column("Interpretation / Context", style="green")

        table.add_row(
            "Minimum Detectable Effect (MDE)",
            f"{self.mde:+.4f}",
            f"Smallest true effect detectable with {self.target_power*100:.0f}% power",
        )
        table.add_row(
            "Standardized MDE (Cohen's d)",
            f"{self.mde_standardized:.3f} σ",
            "Normalized by outcome standard deviation",
        )
        table.add_row("Target Power (1 - β)", f"{self.target_power:.2f}", "Probability of rejecting false null")
        table.add_row("Significance Level (α)", f"{self.alpha:.3f}", "Type I error rate (two-sided)")
        table.add_section()

        for k, v in self.parameters.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            table.add_row(k, val_str, "User/Assumed Specification")

        console.print(table)

    def plot(self) -> Tuple[matplotlib.figure.Figure, Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]:
        """
        Renders a dual-panel power curve:
        Panel 1: Power vs. True Effect Size (highlighting the MDE at target power).
        Panel 2: Power vs. Sample Size / Cluster Count.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=100)

        # --- Panel 1: Power vs. Effect Size ---
        eff_grid = np.linspace(0.0, self.mde * 1.8, 100)
        power_eff = self.curve_func_effect(eff_grid)

        ax1.plot(eff_grid, power_eff, color="#1B4F72", linewidth=2.2, label="Power Curve")
        ax1.axhline(self.target_power, color="#C0392B", linestyle="--", linewidth=1.0, label=f"Target ({self.target_power*100:.0f}%)")
        ax1.axvline(self.mde, color="#16A085", linestyle=":", linewidth=1.4, label=f"MDE = {self.mde:.3f}")
        ax1.scatter([self.mde], [self.target_power], color="#16A085", s=40, zorder=5)

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")
        ax1.set_xlabel("True Effect Size (|Δ|)", fontsize=10, fontweight="medium")
        ax1.set_ylabel("Statistical Power (1 - β)", fontsize=10, fontweight="medium")
        ax1.set_title("Power vs. Effect Size", fontsize=11, fontweight="bold", pad=8)
        ax1.set_ylim(-0.02, 1.02)
        ax1.legend(frameon=False, loc="lower right", fontsize=8.5)

        # --- Panel 2: Power vs. Sample Dimension ---
        s_min = max(4, int(self.sample_param_val * 0.3))
        s_max = int(self.sample_param_val * 2.5)
        sample_grid = np.linspace(s_min, s_max, 100)
        power_sample = self.curve_func_sample(sample_grid)

        ax2.plot(sample_grid, power_sample, color="#2C3E50", linewidth=2.2, label="Power Curve")
        ax2.axhline(self.target_power, color="#C0392B", linestyle="--", linewidth=1.0)
        ax2.axvline(self.sample_param_val, color="#16A085", linestyle=":", linewidth=1.4, label=f"Current ({self.sample_param_val:g})")
        ax2.scatter([self.sample_param_val], [self.target_power], color="#16A085", s=40, zorder=5)

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.2, color="gray")
        ax2.set_xlabel(self.sample_param_name, fontsize=10, fontweight="medium")
        ax2.set_ylabel("Statistical Power (1 - β)", fontsize=10, fontweight="medium")
        ax2.set_title(f"Power vs. {self.sample_param_name}", fontsize=11, fontweight="bold", pad=8)
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(frameon=False, loc="lower right", fontsize=8.5)

        fig.tight_layout()
        return fig, (ax1, ax2)