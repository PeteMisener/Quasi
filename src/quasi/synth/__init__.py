"""Synthetic Control, Augmented SCM, and Synthetic DiD module for quasi."""

from __future__ import annotations

from quasi.synth.ascm import ASCMResult, AugmentedSCM
from quasi.synth.classic import ClassicSCM
from quasi.synth.conformal import ConformalIntervalResult, conformal_sc_interval
from quasi.synth.scm import SCMResult, SyntheticControl
from quasi.synth.sdid import SDiD, SDiDResult, SyntheticDiD

__all__ = [
    "ClassicSCM",
    "SyntheticDiD",
    "SDiD",
    "SyntheticControl",
    "AugmentedSCM",
    "SDiDResult",
    "SCMResult",
    "ASCMResult",
    "ConformalIntervalResult",
    "conformal_sc_interval",
]
