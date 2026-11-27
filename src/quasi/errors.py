"""Typed public errors for design validation and supported inference contracts.

ValueError/NotImplementedError inheritance preserves pre-v1 caller handling.
"""


class QuasiError(Exception):
    """Base exception for Quasi analysis contract failures."""


class InvalidDesignError(QuasiError, ValueError):
    """Data or model specification contradicts the selected design."""


class UnsupportedInferenceError(QuasiError, NotImplementedError):
    """Requested inference is not implemented or independently validated."""


class NotIdentifiedError(InvalidDesignError):
    """Required identifying information or design variation is absent."""


class InsufficientSupportError(InvalidDesignError):
    """Not enough eligible observations or comparison groups."""


class ConvergenceError(QuasiError, RuntimeError):
    """Numerical optimization failed to meet specified convergence checks."""
