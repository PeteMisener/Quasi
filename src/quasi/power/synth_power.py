"""
Synthetic Control and Synthetic DiD Feasibility Audit.
Assesses donor pool dimensions (N0) and pre-intervention periods (T0) to verify
exact permutation test feasibility and prevent donor overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SynthFeasibilityReport:
    """Audit report detailing mathematical viability for Synthetic Control studies."""
    n_donors: int
    n_pre_periods: int
    min_achievable_p_twosided: float
    min_achievable_p_onesided: float
    is_permutation_viable: bool
    is_overfit_risk: bool
    status: str
    warnings: List[str]
    recommendations: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "Synthetic Control / SDiD Ex-Ante Feasibility Audit",
            "=" * 70,
            f"Donor Pool Units (N0):          {self.n_donors}",
            f"Pre-Intervention Periods (T0):  {self.n_pre_periods}",
            f"Min Achievable p-value (2-side): {self.min_achievable_p_twosided:.4f}",
            f"Permutation Test Viability:      {'[PASS] Statistically Viable' if self.is_permutation_viable else '[FAIL] Infeasible (p >= 0.05 guaranteed)'}",
            f"Overfitting Risk (T0 < N0):      {'[WARN] High Overfitting Risk' if self.is_overfit_risk else '[PASS] Low Risk'}",
            "-" * 70,
            "Audit Warnings & Recommendations:",
        ]
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        else:
            lines.append("  (None) Design dimensions satisfy standard empirical criteria.")
        for r in self.recommendations:
            lines.append(f"  [+] {r}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def print_terminal(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="quasi.power: Synthetic Control Feasibility Audit")
        table.add_column("Criterion", style="cyan", no_wrap=True)
        table.add_column("Metric Value", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Guidance & Practical Implications")

        # 1. Permutation viability
        p_status = "[bold green]PASS[/bold green]" if self.is_permutation_viable else "[bold red]FAIL[/bold red]"
        table.add_row(
            "Donor Pool (N0)",
            str(self.n_donors),
            p_status,
            f"Min achievable 2-sided p = {self.min_achievable_p_twosided:.3f} "
            f"({'N0 >= 39 required for p <= 0.05 with ties' if self.n_donors < 39 else 'Adequate donor pool size'})",
        )

        # 2. Time depth
        t_status = "[bold green]PASS[/bold green]" if self.n_pre_periods >= 10 else ("[bold yellow]WARN[/bold yellow]" if self.n_pre_periods >= 5 else "[bold red]FAIL[/bold red]")
        table.add_row(
            "Pre-Periods (T0)",
            str(self.n_pre_periods),
            t_status,
            "T0 >= 10 recommended to establish stable pre-intervention trajectories.",
        )

        # 3. Overfitting risk
        of_status = "[bold yellow]WARN[/bold yellow]" if self.is_overfit_risk else "[bold green]PASS[/bold green]"
        table.add_row(
            "Ratio (T0 / N0)",
            f"{self.n_pre_periods / self.n_donors:.2f}",
            of_status,
            "T0 < N0 can lead to non-unique donor weights interpolating noise. Regularized SDiD or Ridge ASCM required." if self.is_overfit_risk else "Dimension ratio avoids interpolation of random noise.",
        )

        console.print(table)


def synth_feasibility(
    n_donors: int,
    n_pre_periods: int,
    alpha: float = 0.05,
) -> SynthFeasibilityReport:
    """
    Evaluates whether an SCM or SDiD design has sufficient donor and temporal support.

    Parameters:
        n_donors: Number of untreated control donors in pool (N0).
        n_pre_periods: Number of pre-intervention periods (T0).
        alpha: Target false positive rate (default 0.05).
    """
    N0 = n_donors
    T0 = n_pre_periods

    if N0 < 1:
        raise ValueError("Donor pool must have at least 1 donor.")
    if T0 < 2:
        raise ValueError("Must have at least 2 pre-intervention periods.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")

    # Minimum achievable p-value under exact in-space permutation test (Abadie et al.)
    # One-sided: 1 / (N0 + 1)
    # Two-sided: 2 / (N0 + 1)
    min_p_one = 1.0 / (N0 + 1)
    min_p_two = min(1.0, 2.0 / (N0 + 1))

    # A two-sided test at alpha=0.05 CANNOT reject if 2 / (N0 + 1) > 0.05 => N0 < 39
    # A one-sided test CANNOT reject if 1 / (N0 + 1) > 0.05 => N0 < 19
    is_viable = min_p_two <= alpha
    is_overfit = T0 < N0

    warnings: List[str] = []
    recommendations: List[str] = []

    if not is_viable:
        warnings.append(
            f"Small donor pool (N0 = {N0}). The minimum achievable two-sided p-value in an in-space "
            f"permutation test is {min_p_two:.3f} > α ({alpha}). You cannot reject the null hypothesis of no effect."
        )
        recommendations.append("Expand donor pool to at least N0 = 39 donors, or test at one-sided α = 0.05 (requires N0 >= 19).")
    elif N0 < 39:
        warnings.append(f"Donor pool N0 = {N0} is marginal. Exactly zero donor placebos can exceed the treated effect to achieve p < 0.05.")
        recommendations.append("Aim for N0 >= 40 donors to allow for placebo variability.")

    if T0 < 5:
        warnings.append(f"Short pre-intervention history (T0 = {T0}). Pre-treatment match quality will be unreliable.")
        recommendations.append("Gather higher-frequency data (e.g. monthly instead of yearly) to expand T0.")

    if is_overfit:
        warnings.append(
            f"T0 ({T0}) < N0 ({N0}). The donor optimization is underdetermined, meaning classic SCM can perfectly "
            "interpolate idiosyncratic noise rather than genuine common factors."
        )
        recommendations.append("Use Synthetic DiD (quasi.synth.SyntheticDiD) or Augmented SCM with L2 ridge regularization.")

    status = "VIABLE" if is_viable and not is_overfit else ("VIABLE_WITH_WARNINGS" if is_viable else "INFEASIBLE")

    return SynthFeasibilityReport(
        n_donors=N0,
        n_pre_periods=T0,
        min_achievable_p_twosided=min_p_two,
        min_achievable_p_onesided=min_p_one,
        is_permutation_viable=is_viable,
        is_overfit_risk=is_overfit,
        status=status,
        warnings=warnings,
        recommendations=recommendations,
    )