"""Exploratory RD continuity and mass-point screens; not identification tests.

Covariate p-values use Holm family-wise adjustment, but covariate balance
cannot prove continuity of unobserved potential outcomes. Density/discreteness
screens are descriptive and not replacements for rddensity.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np
from scipy import stats
from quasi.base import DiagnosticStatus, DiagnosticTest
from quasi.rdd.kernels import KERNEL_REGISTRY
from quasi.rdd.local_inference import rd_inference


def support_screen(x: np.ndarray, bandwidth: float, kernel: str, order: int) -> DiagnosticTest:
    """Descriptive running-variable effective-support check, not a p-value."""
    weight = KERNEL_REGISTRY[kernel](x / bandwidth) > 0
    left = x[(x < 0) & weight]
    right = x[(x >= 0) & weight]
    n_left, n_right = len(left), len(right)
    u_left, u_right = np.unique(left).size, np.unique(right).size
    min_unique = min(u_left, u_right)
    dense = (min_unique < max(10, order + 2))
    share_left = u_left / n_left if n_left else 0
    share_right = u_right / n_right if n_right else 0
    substantial_ties = min(share_left, share_right) < 0.5
    return DiagnosticTest(
        name="Running-variable effective support (descriptive)",
        status=DiagnosticStatus.WARN if dense or substantial_ties else DiagnosticStatus.PASS,
        statistic_name="Minimum distinct cutoff-side scores",
        statistic_value=float(min_unique), p_value=None,
        threshold="descriptive: >=10 distinct scores/side and >=50% distinct ratio",
        message=(f"Within kernel support: left {n_left} obs/{u_left} unique scores; "
                 f"right {n_right} obs/{u_right} unique scores. "
                 "These descriptive thresholds do not validate coverage or justify mass-point adjustments."),
        remedy="Investigate heaping, measurement precision, and discrete-score inference" if dense or substantial_ties else None,
    )


def covariate_continuity(x: np.ndarray, values: Mapping[str, np.ndarray],
                         bandwidth: float, kernel: str, order: int, *, alpha: float = 0.05
                         ) -> dict[str, DiagnosticTest]:
    """One-sided RD placebo contrasts with Holm-adjusted p-values.

    All covariates must be measured prior to treatment. If the same covariate is
    checked at several bandwidths the family should be adjusted across *all*
    planned tests; this helper adjusts only the covariates in one specification.
    """
    raw = []
    names = list(values)
    unavailable = set()
    for name in names:
        array = np.asarray(values[name], dtype=float)
        if float(np.ptp(array)) <= 1e-12 * max(1.0, float(np.max(np.abs(array)))):
            unavailable.add(name)
            raw.append((0.0, 0.0, 1.0))
            continue
        try:
            effect = rd_inference(x, array, bandwidth, kernel, order)
        except ValueError as exc:
            # A constant or under-supported covariate must not destroy a valid
            # outcome fit. It is an *unavailable diagnostic*, not a PASS.
            unavailable.add(name)
            raw.append((0.0, 0.0, 1.0))
            continue
        z = effect['estimate'] / effect['se']
        p = float(2 * stats.norm.sf(abs(z)))
        raw.append((effect['estimate'], z, p))
    # Holm step-down adjustment, monotone under sorted p-values.
    adjusted = np.empty(len(raw), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(np.argsort([item[2] for item in raw])):
        running_max = max(running_max, (len(raw) - rank) * raw[index][2])
        adjusted[index] = min(1.0, running_max)
    output = {}
    for index, name in enumerate(names):
        jump, z, p = raw[index]
        flagged = adjusted[index] < alpha
        if name in unavailable:
            output[f"Covariate continuity: {name}"] = DiagnosticTest(
                name=f"Predetermined covariate continuity: {name} (unavailable)",
                status=DiagnosticStatus.WARN, statistic_name="Not estimable",
                statistic_value=0.0, p_value=None, threshold="no statistical test",
                message="Covariate RD variance or local support was insufficient; no continuity test performed.",
                remedy="Examine scale, constancy and support of the predetermined covariate",
            )
            continue
        output[f"Covariate continuity: {name}"] = DiagnosticTest(
            name=f"Predetermined covariate continuity: {name} (exploratory)",
            status=DiagnosticStatus.WARN if flagged else DiagnosticStatus.PASS,
            statistic_name="Local discontinuity z (HC0)", statistic_value=float(z),
            p_value=float(adjusted[index]),
            threshold=f"Holm-adjusted p < {alpha:g} flags possible imbalance",
            message=(f"Estimated covariate discontinuity {jump:.5g}; raw p={p:.4g}, "
                     f"Holm p={adjusted[index]:.4g}. "
                     "Non-rejection does NOT demonstrate covariate continuity or causal identification."),
            remedy="Review assignment mechanisms and pre-treatment covariate measurement" if flagged else None,
        )
    return output
