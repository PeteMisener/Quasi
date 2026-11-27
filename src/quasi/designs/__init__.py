"""Explicit, immutable design specifications and fail-closed data checks."""
from quasi.designs.specs import (
    DesignSpec, PanelDesign, RDDDesign, IVDesign, ITSDesign, SyntheticDesign, ObservationalDesign,
)
from quasi.designs.validation import DesignAudit, validate_design_data

__all__ = [
    "DesignSpec", "PanelDesign", "RDDDesign", "IVDesign", "ITSDesign",
    "SyntheticDesign", "ObservationalDesign", "DesignAudit", "validate_design_data",
]
