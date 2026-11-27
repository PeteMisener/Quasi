"""Backward-compatible simplex projection & QP shim.

Canonical implementations now live in ``quasi.inference.optimization``.
"""

from __future__ import annotations

import numpy as np

from quasi.inference.optimization import checked_simplex_qp, project_simplex


def solve_simplex_qp(
    Q: np.ndarray,
    p: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Solve min 0.5 w'Qw - p'w over the unit simplex via checked FISTA."""
    return checked_simplex_qp(Q, p, max_iter=max_iter, tol=tol).weights


__all__ = ["project_simplex", "solve_simplex_qp"]
