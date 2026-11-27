"""Classic Abadie et al. Synthetic Control Method."""

from __future__ import annotations

import numpy as np

from quasi.synth.scm import _fit_scm_weights


class ClassicSCM:
    """Abadie, Diamond, and Hainmueller (2010) Synthetic Control Optimizer."""

    def __init__(
        self,
        Y_pre_trt: np.ndarray,      # shape (T_pre,)
        Y_pre_ctrl: np.ndarray,     # shape (N_ctrl, T_pre)
    ) -> None:
        self.Y_pre_trt = Y_pre_trt
        self.Y_pre_ctrl = Y_pre_ctrl
        self.N_ctrl, self.T_pre = Y_pre_ctrl.shape

    def fit_weights(self) -> np.ndarray:
        """Fit an outcome-only classic SCM simplex; refuse unconverged solutions."""
        target = np.asarray(self.Y_pre_trt, dtype=float)
        controls = np.asarray(self.Y_pre_ctrl, dtype=float)
        if target.shape != (self.T_pre,) or controls.ndim != 2:
            raise ValueError("SCM pre-treatment outcome shapes must agree")
        return _fit_scm_weights(target, controls)
