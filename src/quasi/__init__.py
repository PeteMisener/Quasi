"""
quasi: High-precision, human-centered Quasi-Experimental Design in Python.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from quasi import ai
from quasi.designs import (DesignSpec, DesignAudit, PanelDesign, RDDDesign, IVDesign, ITSDesign, SyntheticDesign, ObservationalDesign, validate_design_data)
from quasi.errors import (QuasiError, InvalidDesignError, UnsupportedInferenceError, NotIdentifiedError, InsufficientSupportError, ConvergenceError)
from quasi.base import BaseQEDModel, DiagnosticStatus, DiagnosticTest, QEDResult
from quasi.did.sensitivity import HonestDiDResult, honest_did_sensitivity
from quasi.did.staggered import StaggeredDiD, StaggeredDiDResult
from quasi.diagnostics.density import cjm_density_test
from quasi.its.controlled import CITSResult, ControlledITS
from quasi.its.segmented import ITSResult, InterruptedTimeSeries
from quasi.iv.tsls import IVResult, TwoStageLS
from quasi.matching.psm import MatchingResult, PropensityScoreMatching, AIPWATT, IPWATT
from quasi.observational.sensitivity import OVBSensitivityResult, ovb_sensitivity
from quasi.power.base import PowerResult
from quasi.power.did_power import did_mde, did_sample_size
from quasi.power.rdd_power import rdd_mde, rdd_sample_size
from quasi.power.simulation import SimulationPowerReport, simulate_power
from quasi.power.synth_power import SynthFeasibilityReport, synth_feasibility
from quasi.prep.audit import audit_panel
from quasi.rdd.regression import Discontinuity, FuzzyRDD, RDDResult, SharpRDD
from quasi.reporting.table import ModelComparisonTable, TableFormattingOptions, compare_models
from quasi.results import ResultRecord, InferenceRecord, SampleFlow, AssumptionRecord, DiagnosticRecord, ProvenanceRecord
from quasi.router import estimate
from quasi.synth.ascm import ASCMResult, AugmentedSCM
from quasi.synth.conformal import ConformalIntervalResult, conformal_sc_interval
from quasi.synth.sdid import SDiD, SDiDResult, SyntheticDiD
from quasi.synth.scm import SyntheticControl, SCMResult

try:
    __version__ = version("quasi-qed")
except PackageNotFoundError:
    # Source checkout without an installed distribution.
    __version__ = "0+unknown"


def ask(query: str) -> str:
    """Ask an econometric design question to the quasi assistant."""
    from quasi.ai.router import ask_assistant
    return ask_assistant(query)


__all__ = [
    "estimate",
    "ResultRecord", "InferenceRecord", "SampleFlow", "AssumptionRecord", "DiagnosticRecord", "ProvenanceRecord",
    "DesignSpec", "DesignAudit", "PanelDesign", "RDDDesign", "IVDesign", "ITSDesign",
    "SyntheticDesign", "ObservationalDesign", "validate_design_data",
    "QuasiError", "InvalidDesignError", "UnsupportedInferenceError",
    "NotIdentifiedError", "InsufficientSupportError", "ConvergenceError",
    "ask",
    "audit_panel",
    "compare_models",
    "ModelComparisonTable",
    "TableFormattingOptions",
    "BaseQEDModel",
    "QEDResult",
    "DiagnosticTest",
    "DiagnosticStatus",
    "ai",
    # DiD
    "StaggeredDiD",
    "StaggeredDiDResult",
    "HonestDiDResult",
    "honest_did_sensitivity",
    # RDD
    "Discontinuity",
    "SharpRDD",
    "FuzzyRDD",
    "RDDResult",
    "cjm_density_test",
    # SCM / SDiD
    "SyntheticDiD",
    "SDiD",
    "SyntheticControl",
    "SCMResult",
    "SDiDResult",
    "AugmentedSCM",
    "ASCMResult",
    "ConformalIntervalResult",
    "conformal_sc_interval",
    # ITS
    "InterruptedTimeSeries",
    "ITSResult",
    "ControlledITS",
    "CITSResult",
    # IV
    "TwoStageLS",
    "IVResult",
    # Observational & Sensitivity
    "PropensityScoreMatching",
    "MatchingResult",
    "AIPWATT",
    "IPWATT",
    "OVBSensitivityResult",
    "ovb_sensitivity",
    # Power & MDE
    "did_mde",
    "did_sample_size",
    "rdd_mde",
    "rdd_sample_size",
    "synth_feasibility",
    "simulate_power",
    "PowerResult",
    "SimulationPowerReport",
    "SynthFeasibilityReport",
]