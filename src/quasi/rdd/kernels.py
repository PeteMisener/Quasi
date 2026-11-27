"""Kernel weighting functions and asymptotic moment constants for local polynomials."""

from __future__ import annotations

from typing import Callable, Dict
import numpy as np


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Boundary MSE-optimal triangular kernel: K(u) = (1 - |u|) * I(|u| <= 1)."""
    abs_u = np.abs(u)
    return np.where(abs_u <= 1.0, 1.0 - abs_u, 0.0)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov parabolic kernel: K(u) = 0.75 * (1 - u^2) * I(|u| <= 1)."""
    abs_u = np.abs(u)
    return np.where(abs_u <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform boxcar kernel: K(u) = 0.5 * I(|u| <= 1)."""
    abs_u = np.abs(u)
    return np.where(abs_u <= 1.0, 0.5, 0.0)


KERNEL_REGISTRY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "triangular": triangular_kernel,
    "epanechnikov": epanechnikov_kernel,
    "uniform": uniform_kernel,
}

# Kernel constants: C_K = (int K^2 / (4 * (int u^2 K)^2))^(1/5)
KERNEL_C_K: Dict[str, float] = {
    "triangular": 1.43097,   # ( (2/3) / (4 * (1/6)^2) )^(1/5) = 6^(1/5)
    "epanechnikov": 1.41920, # ( (3/5) / (4 * (1/5)^2) )^(1/5) = (15/4)^(1/5)
    "uniform": 1.35100,      # ( (1/2) / (4 * (1/3)^2) )^(1/5) = (9/8)^(1/5)
}