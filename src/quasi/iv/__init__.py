"""Instrumental Variables (2SLS) module for quasi."""

from __future__ import annotations

from quasi.iv.tsls import IVResult, TwoStageLS

__all__ = [
    "TwoStageLS",
    "IVResult",
]