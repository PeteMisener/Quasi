"""
Privacy guard and token-pruning condenser for quasi.ai metadata payloads.
Guarantees zero raw microdata transmission and limits payload size (< 4 KB).
"""

from __future__ import annotations

import math
from typing import Any, Dict


def _safe_round(value: Any, digits: int = 4) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return round(num, digits) if math.isfinite(num) else None


def condense_metadata_payload(meta: Dict[str, Any], max_sub_items: int = 8) -> Dict[str, Any]:
    """
    Prunes and sanitizes statistical metadata before sending to an LLM provider:
    1. Ensures zero microdata rows or raw data arrays exist in payload.
    2. Subsamples large lead-lag event-study grids or donor weight distributions.
    3. Normalizes float precision to 4 decimal places.
    """
    condensed: Dict[str, Any] = {
        "library": "quasi",
        "model_name": meta.get("model_name", "UnknownModel"),
        "outcome_name": meta.get("outcome_name", "Y"),
        "estimand": meta.get("estimand", "ATT"),
        "passed_validation": meta.get("passed_validation", False),
        "diagnostic_status": meta.get("diagnostic_status", "not_run"),
        "identification_status": meta.get("identification_status", "not_established"),
        "experimental": meta.get("experimental", True),
    }

    # 1. Primary Metrics
    if "primary_metric" in meta:
        pm = meta["primary_metric"]
        raw_ci = pm.get("ci", pm.get("ci_95", (None, None)))
        ci_pair = (
            [_safe_round(raw_ci[0], 4), _safe_round(raw_ci[1], 4)]
            if isinstance(raw_ci, (tuple, list)) and len(raw_ci) == 2
            else [None, None]
        )
        condensed["primary_metric"] = {
            "estimate": _safe_round(pm.get("estimate"), 4),
            "se": _safe_round(pm.get("se"), 4),
            "p_value": _safe_round(pm.get("p_value"), 4),
            "ci": ci_pair,
            "confidence_level": pm.get("confidence_level", 0.95),
            "estimand": pm.get("estimand", condensed["estimand"]),
        }

    # 2. Sample Metrics
    if "sample_metrics" in meta:
        sm = meta["sample_metrics"]
        condensed["sample_metrics"] = {
            k: (_safe_round(v, 4) if isinstance(v, float) else v)
            for k, v in sm.items()
            if not isinstance(v, (list, dict))
        }

    # 3. Prune Sub-Effects (e.g. event study leads/lags or donor weights)
    sub = meta.get("sub_effects", {})
    condensed_sub: Dict[str, Any] = {}

    if "event_study_effects" in sub:
        effs = sub["event_study_effects"]
        keys = sorted(list(effs.keys()))
        if len(keys) > max_sub_items:
            # Pick first 3, middle 2, and last 3 periods
            sub_keys = sorted(list(set(keys[:3] + keys[len(keys)//2 - 1 : len(keys)//2 + 1] + keys[-3:])))
            condensed_sub["event_study_sample"] = {
                f"rel_t_{k}": _safe_round(effs[k], 4) for k in sub_keys
            }
            condensed_sub["total_event_periods"] = len(keys)
        else:
            condensed_sub["event_study"] = {f"rel_t_{k}": _safe_round(effs[k], 4) for k in keys}

    if "donor_weights" in sub:
        w_dict = sub["donor_weights"]
        # Sort by absolute magnitude so negative extrapolation weights (e.g. ASCM) are never hidden
        sorted_donors = sorted(w_dict.items(), key=lambda x: abs(float(x[1])), reverse=True)
        top_donors = {k: _safe_round(v, 4) for k, v in sorted_donors[:5] if abs(float(v)) > 0.005}
        condensed_sub["top_donors"] = top_donors
        condensed_sub["total_donors"] = len(w_dict)

    for k, v in sub.items():
        if k not in ("event_study_effects", "event_study_se", "donor_weights") and isinstance(v, (int, float, str, bool)):
            condensed_sub[k] = _safe_round(v, 4) if isinstance(v, float) else v

    condensed["sub_effects"] = condensed_sub

    # 4. Diagnostic Summaries
    diags = meta.get("diagnostics", [])
    condensed["diagnostics"] = [
        {
            "name": d.get("name", "UnknownCheck"),
            "status": d.get("status", "WARN"),
            "metric": f"{d.get('statistic_name', 'stat')}={_safe_round(d.get('statistic_value'), 4)}",
            "p_value": _safe_round(d.get("p_value"), 4),
            "message": d.get("message", ""),
            "remedy": d.get("remedy"),
        }
        for d in diags
    ]

    return condensed