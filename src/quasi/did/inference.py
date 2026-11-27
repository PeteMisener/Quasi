"""Experimental influence-function aggregation for balanced panel DiD.

The influence function is defined at the panel-unit level. If ``cluster`` is
specified, sums of unit influence functions are the independent sampling
pieces; no observations (time rows) are resampled independently.

These are *conditional-on-observed-cohort-shares* variance approximations,
not inference certified equivalent to R ``did`` or suitable with few clusters.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class InferencePlan:
    labels: np.ndarray | None
    n_units: int
    n_clusters: int

    @classmethod
    def build(cls, labels: np.ndarray | None, n_units: int) -> "InferencePlan":
        if labels is None:
            return cls(None, n_units, n_units)
        labels = np.asarray(labels, dtype=int)
        if labels.shape != (n_units,) or np.any(labels < 0):
            raise ValueError("Cluster codes must identify every panel unit")
        m = int(np.unique(labels).size)
        if m < 10:
            raise ValueError("Clustered DiD requires at least 10 distinct clusters; few-cluster inference is unsupported")
        return cls(labels, n_units, m)

    @property
    def correction(self) -> float:
        # The per-cluster finite-count scaling is explicit, without claiming
        # equivalence to design-specific regression degrees-of-freedom rules.
        return self.n_clusters / (self.n_clusters - 1) if self.labels is not None else 1.0

    def scores(self, psi: np.ndarray) -> np.ndarray:
        """Return independent influence contributions, one row per cluster/unit."""
        arr = np.asarray(psi, dtype=float)
        if arr.shape[0] != self.n_units or arr.ndim not in (1, 2):
            raise ValueError("Influence must have one row per panel unit")
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite influence values")
        if self.labels is None:
            return arr
        if arr.ndim == 1:
            return np.bincount(self.labels, weights=arr, minlength=self.n_clusters)
        result = np.zeros((self.n_clusters, arr.shape[1]), dtype=float)
        np.add.at(result, self.labels, arr)
        return result

    def covariance(self, influence: np.ndarray) -> np.ndarray:
        scores = self.scores(influence)
        if scores.ndim == 1:
            scores = scores[:, None]
        return self.correction * (scores.T @ scores) / (self.n_units ** 2)

    def se(self, influence: np.ndarray) -> float:
        return float(np.sqrt(max(0.0, self.covariance(influence)[0, 0])))

    def tail_probability(self, statistic: float) -> float:
        if self.labels is not None:
            return float(2 * stats.t.sf(abs(statistic), df=self.n_clusters - 1))
        return float(2 * stats.norm.sf(abs(statistic)))

    def critical(self, alpha: float) -> float:
        if self.labels is not None:
            return float(stats.t.ppf(1 - alpha / 2, df=self.n_clusters - 1))
        return float(stats.norm.ppf(1 - alpha / 2))

    def multiplier_max_t(self, influence: np.ndarray, se: np.ndarray, *,
                         reps: int, seed: int, alpha: float) -> float:
        scores = self.scores(influence)
        if scores.ndim != 2 or scores.shape[1] != len(se):
            raise ValueError("Multiplier influences and standard errors do not align")
        rng = np.random.default_rng(seed)
        maxima = np.empty(reps)
        correction = np.sqrt(self.correction)
        for begin in range(0, reps, 256):
            stop = min(begin + 256, reps)
            multipliers = rng.choice([-1.0, 1.0], size=(stop - begin, self.n_clusters))
            numer = multipliers @ scores * correction / self.n_units
            maxima[begin:stop] = np.max(np.abs(numer / se), axis=1)
        return float(np.quantile(maxima, 1 - alpha))
