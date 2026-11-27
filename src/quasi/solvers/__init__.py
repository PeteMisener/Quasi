"""Backward-compatible simplex solver exports."""

from quasi.solvers.apgd import project_simplex, solve_simplex_qp

__all__ = ["project_simplex", "solve_simplex_qp"]
