"""Automated BibTeX citation and deterministic plain-English text generator."""

from __future__ import annotations

import math
from typing import Any, Dict

CITATIONS: Dict[str, str] = {
    "CallawaySantAnnaDiD": """@article{callaway2021difference,
  title={Difference-in-differences with multiple time periods},
  author={Callaway, Brantly and Sant'Anna, Pedro HC},
  journal={Journal of Econometrics},
  volume={225},
  number={2},
  pages={200--230},
  year={2021}
}""",
    "SharpRDD": """@article{calonico2014robust,
  title={Robust nonparametric confidence intervals for regression-discontinuity designs},
  author={Calonico, Sebastian and Cattaneo, Matias D and Titiunik, Rocio},
  journal={Econometrica},
  volume={82},
  number={6},
  pages={2295--2326},
  year={2014}
}""",
    "FuzzyRDD": """@article{calonico2014robust,
  title={Robust nonparametric confidence intervals for regression-discontinuity designs},
  author={Calonico, Sebastian and Cattaneo, Matias D and Titiunik, Rocio},
  journal={Econometrica},
  volume={82},
  number={6},
  pages={2295--2326},
  year={2014}
}""",
    "SyntheticDiD": """@article{arkhangelsky2021synthetic,
  title={Synthetic difference-in-differences},
  author={Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David A and Imbens, Guido W and Wager, Stefan},
  journal={American Economic Review},
  volume={111},
  number={12},
  pages={4088--4118},
  year={2021}
}""",
    "SyntheticControl": """@article{abadie2010synthetic,
  title={Synthetic control methods for comparative case studies},
  author={Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},
  journal={Journal of the American Statistical Association},
  volume={105},
  number={490},
  pages={493--505},
  year={2010}
}""",
}


def generate_bibtex(model_name: str) -> str:
    """Return BibTeX citation for given model."""
    return CITATIONS.get(model_name, "@misc{quasi2026, title={quasi: Python QED Modeling}}")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def generate_deterministic_explanation(meta: Dict[str, Any], audience: str = "academic") -> str:
    """Deterministic, template-based explanation generator requiring zero network calls."""
    pm = meta["primary_metric"]
    est = float(pm["estimate"])
    se = pm.get("se")
    p = pm.get("p_value")
    outcome = meta["outcome_name"]
    model = meta["model_name"]

    confidence = float(pm.get("confidence_level", 0.95))
    alpha = 1.0 - confidence
    has_inference = _is_finite_number(se) and _is_finite_number(p)

    if not has_inference:
        if audience == "academic":
            return (
                f"Point estimation conducted via {model} yields an estimated effect on {outcome} "
                f"of {est:.4f}. Calibrated sampling uncertainty (SE, p-value, and CI) is not available "
                f"for this estimator. Diagnostic check status: {meta.get('diagnostic_status', 'not_run')}. "
                "Causal identification depends on assumptions not established by diagnostics alone."
            )
        direction = "increase" if est > 0 else "decrease"
        return (
            f"The model estimates a {abs(est):.2f} unit {direction} in {outcome}. "
            "Formal statistical uncertainty intervals are unavailable for this method; "
            "causal interpretation requires unverified design assumptions."
        )

    se_f = float(se)
    p_f = float(p)
    sig = (
        "statistically significant at the specified threshold"
        if p_f < alpha
        else "not statistically distinguishable from zero at the specified threshold"
    )

    if audience == "academic":
        return (
            f"Estimation conducted via {model} indicates a {sig} treatment effect on {outcome} "
            f"(estimate = {est:.4f}, SE = {se_f:.4f}, p = {p_f:.4f}). "
            f"Diagnostic check status: {meta.get('diagnostic_status', 'not_run')}. "
            "Causal identification depends on assumptions not established by diagnostics alone."
        )

    direction = "increase" if est > 0 else "decrease"
    return (
        f"The model estimates a {abs(est):.2f} unit {direction} in {outcome}. "
        f"The estimate is {sig} (p = {p_f:.3f}); causal interpretation requires unverified design assumptions."
    )
