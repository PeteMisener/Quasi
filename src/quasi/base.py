"""Core architectural contracts, base classes, and protocols for quasi."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from quasi.prep.extractor import extract_column_arrays

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


class DiagnosticStatus(str, Enum):
    """Execution status for an econometric identification or assumption check."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DiagnosticTest:
    """Individual econometric validity or falsification test result."""
    name: str
    status: DiagnosticStatus
    statistic_name: str
    statistic_value: float
    p_value: Optional[float]
    threshold: str
    message: str
    remedy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "statistic_name": self.statistic_name,
            "statistic_value": self.statistic_value,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "message": self.message,
            "remedy": self.remedy,
        }


class QEDResult(ABC):
    """Standardized result object returned by all models in quasi."""

    def __init__(
        self,
        estimate: float,
        se: float,
        p_value: float,
        ci: Tuple[float, float],
        estimand: str,
        outcome_name: str,
        model_name: str,
        n_obs: int,
        n_treated: int,
        diagnostics: Optional[Dict[str, DiagnosticTest]] = None,
        sub_effects: Optional[Dict[str, Any]] = None,
        model_meta: Optional[Dict[str, Any]] = None,
        underlying_model: Optional[BaseQEDModel] = None,
    ) -> None:
        self.estimate = float(estimate)
        self.se = float(se)
        self.p_value = float(p_value)
        self.ci_lower, self.ci_upper = float(ci[0]), float(ci[1])
        self.estimand = estimand
        self.outcome_name = outcome_name
        self.model_name = model_name
        self.n_obs = n_obs
        self.n_treated = n_treated
        self.diagnostics = diagnostics or {}
        self.sub_effects = sub_effects or {}
        self.model_meta = model_meta or {}
        self._model = underlying_model
        self.confidence_level = (
            1.0 - underlying_model.alpha if underlying_model is not None else 0.95
        )

    @property
    def passed_validation(self) -> bool:
        """Backward-compatible flag: every *executed* diagnostic is PASS.

        This never establishes causal identification. Empty and warning-only
        diagnostic suites do not pass. Prefer ``identification_status``.
        """
        return bool(self.diagnostics) and all(
            t.status == DiagnosticStatus.PASS for t in self.diagnostics.values()
        )

    @property
    def identification_status(self) -> str:
        """Diagnostic tests cannot establish identifying assumptions from data."""
        return "not_established"

    @property
    def diagnostic_status(self) -> str:
        if not self.diagnostics:
            return "not_run"
        if any(t.status == DiagnosticStatus.FAIL for t in self.diagnostics.values()):
            return "failed"
        if any(t.status == DiagnosticStatus.WARN for t in self.diagnostics.values()):
            return "warnings"
        return "passed_checks_only"

    @abstractmethod
    def summary(self) -> str:
        """Return formatted console summary table."""
        pass

    @abstractmethod
    def plot(self, **kwargs: Any) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Generate standardized publication visualization."""
        pass

    @abstractmethod
    def bibtex(self) -> str:
        """Return BibTeX citation for estimator and standard error procedures."""
        pass

    @abstractmethod
    def methods_text(self) -> str:
        """Return publication-ready methodology paragraph."""
        pass

    def validate(self) -> Dict[str, DiagnosticTest]:
        """Display diagnostic audit table in terminal and return dictionary."""
        self._render_diagnostic_console()
        return self.diagnostics

    def robustness(self, **kwargs: Any) -> Any:
        """Execute one-call sensitivity sweep across bandwidths, kernels, or specifications."""
        raise NotImplementedError(f".robustness() is not implemented for {self.model_name}")

    def explain(
        self,
        audience: Literal["academic", "stakeholder", "student"] = "academic",
        engine: Literal["rule_based", "ai"] = "rule_based",
    ) -> str:
        """Generate plain-English interpretation of findings."""
        if engine == "rule_based":
            from quasi.reporting.provenance import generate_deterministic_explanation
            return generate_deterministic_explanation(self.to_metadata_dict(), audience=audience)

        try:
            from quasi.ai.router import route_explanation_request
            return route_explanation_request(self.to_metadata_dict(), audience=audience)
        except ImportError:
            return (
                "QuasiAIError: AI assistant dependencies not found. "
                "Install using: pip install quasi-qed[ai]"
            )

    def diagnose(self) -> str:
        """AI-powered troubleshooting of failed assumption checks."""
        try:
            from quasi.ai.router import route_diagnosis_request
            return route_diagnosis_request(self.to_metadata_dict())
        except ImportError:
            return (
                "QuasiAIError: AI assistant dependencies not found. "
                "Install using: pip install quasi-qed[ai]"
            )

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Serialize statistical results into a privacy-safe JSON dictionary."""
        return {
            "library": "quasi",
            "model_name": self.model_name,
            "outcome_name": self.outcome_name,
            "estimand": self.estimand,
            "sample_metrics": {
                "n_obs": self.n_obs,
                "n_treated": self.n_treated,
                **self.model_meta.get("sample_metrics", {}),
            },
            "primary_metric": {
                "estimate": self.estimate,
                "se": self.se,
                "p_value": self.p_value,
                "ci": (self.ci_lower, self.ci_upper),
                "confidence_level": self.confidence_level,
                # Legacy field only when it genuinely denotes a 95% interval.
                **({"ci_95": (self.ci_lower, self.ci_upper)} if np.isclose(self.confidence_level, 0.95) else {}),
                "estimand": self.estimand,
            },
            "sub_effects": self.sub_effects,
            "diagnostics": [test.to_dict() for test in self.diagnostics.values()],
            "passed_validation": self.passed_validation,  # executed diagnostics only; deprecated
            "diagnostic_status": self.diagnostic_status,
            "identification_status": self.identification_status,
            "experimental": True,
            "schema_version": "1.0-draft",
            "design_audit": getattr(self, "_design_audit", None).to_dict() if getattr(self, "_design_audit", None) is not None else None,
            "inference_validation": "pending",
            "inference_method": getattr(self, "inference_details", {}).get("method", "unverified_estimator_specific"),
            "inference_settings": getattr(self, "inference_details", {"reference_certified": False}),
        }

    def to_record(self):
        """Return a frozen, versioned evidence record (experimental schema).

        Direct constructor usage lacks a design audit and is labeled as such.
        Neither a record nor passed diagnostics certifies identification.
        """
        from quasi.results.schema import from_estimator_result
        return from_estimator_result(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize reproducible metadata, excluding input data and model internals."""
        return self.to_record().to_json(indent=indent)

    def to_frame(self) -> pd.DataFrame:
        """One-row statistical summary; CI confidence is explicit."""
        return pd.DataFrame([{
            "model": self.model_name,
            "outcome": self.outcome_name,
            "estimand": self.estimand,
            "estimate": self.estimate,
            "se": self.se,
            "p_value": self.p_value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "identification_status": self.identification_status,
        }])

    def diagnostic_table(self) -> pd.DataFrame:
        """No terminal dependencies; respects the existing .diagnostics dictionary."""
        return pd.DataFrame([t.to_dict() for t in self.diagnostics.values()])

    def _repr_html_(self) -> str:
        """Interactive rich HTML representation for Jupyter notebooks."""
        from quasi.reporting.html_card import generate_result_html
        return generate_result_html(self)

    def _render_diagnostic_console(self) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print("[bold yellow]EXPERIMENTAL: reference statistical validation pending; diagnostic checks are not proof of identification.[/bold yellow]")
        table = Table(title=f"Quasi Exploratory Diagnostics: {self.model_name}")
        table.add_column("Test / Gate", style="cyan", no_wrap=True)
        table.add_column("Statistic", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Threshold", justify="center")
        table.add_column("Status", justify="center")
        table.add_column("Details & Recommendations")

        for test in self.diagnostics.values():
            if test.status == DiagnosticStatus.PASS:
                status_str = "[bold green]PASS[/bold green]"
            elif test.status == DiagnosticStatus.WARN:
                status_str = "[bold yellow]WARN[/bold yellow]"
            else:
                status_str = "[bold red]FAIL[/bold red]"

            stat_display = f"{test.statistic_name}={test.statistic_value:.4f}"
            p_display = f"{test.p_value:.4f}" if test.p_value is not None else "-"
            details = test.message + (f" [bold]Remedy:[/bold] {test.remedy}" if test.remedy else "")

            table.add_row(test.name, stat_display, p_display, test.threshold, status_str, details)

        console.print(table)


class BaseQEDModel(ABC):
    """
    Abstract base class for all quasi-experimental models.
    Enforces zero-copy ingestion, automatic auditing, and fit() protocol.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, Any],
        outcome: str,
        covariates: Optional[List[str]] = None,
        cluster: Optional[str] = None,
        alpha: float = 0.05,
    ) -> None:
        self.outcome = outcome
        self.covariates = covariates or []
        self.cluster = cluster
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        self.alpha = alpha

        # Memory-safe zero-copy extraction
        col_map: Dict[str, Union[str, List[str], None]] = {
            "outcome": self.outcome,
            "covariates": self.covariates if self.covariates else None,
            "cluster": self.cluster,
        }
        self.arrays = extract_column_arrays(data, col_map)
        self.n_rows = len(self.arrays["outcome"])

    @abstractmethod
    def fit(self) -> QEDResult:
        """Run estimation and return a standardized QEDResult."""
        pass