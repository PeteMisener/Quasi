"""Ex-ante power analysis and minimum detectable effect (MDE) module for quasi."""

from __future__ import annotations

from quasi.power.base import PowerResult
from quasi.power.did_power import did_mde, did_sample_size
from quasi.power.rdd_power import rdd_mde, rdd_sample_size
from quasi.power.simulation import SimulationPowerReport, simulate_power
from quasi.power.synth_power import SynthFeasibilityReport, synth_feasibility

__all__ = [
    "PowerResult",
    "SimulationPowerReport",
    "SynthFeasibilityReport",
    "did_mde",
    "did_sample_size",
    "rdd_mde",
    "rdd_sample_size",
    "simulate_power",
    "synth_feasibility",
]