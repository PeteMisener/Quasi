"""Difference-in-Differences (DiD) module for quasi."""

from __future__ import annotations

from quasi.did.staggered import StaggeredDiD, StaggeredDiDResult

# Ergonomic aliases matching different econometric and user naming conventions
Staggered = StaggeredDiD
CallawaySantAnna = StaggeredDiD
CallawaySantAnnaDiD = StaggeredDiD

__all__ = [
    "StaggeredDiD",
    "StaggeredDiDResult",
    "Staggered",
    "CallawaySantAnna",
    "CallawaySantAnnaDiD",
]