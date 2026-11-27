"""Compare identical frozen donor-bootstrap draws to an *executed* R synthdid run.

This runner NEVER generates R results; authentic R output must be supplied.
The benchmark compares numerical effects and resampling variance. Even perfect
agreement does not certify the variance's sampling coverage or causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quasi.synth.sdid import _sdid_weights, _sdid_effect
from validation.sdid_phase10b_reference import FIXTURES, fixture_inputs

DRAWS = FIXTURES / "sdid_phase10c_draws.json"
DRAWS_HASH = FIXTURES / "sdid_phase10c_draws.sha256"


def read_draws() -> tuple[dict, list[list[int]], str]:
    binary = DRAWS.read_bytes()
    checksum = hashlib.sha256(binary).hexdigest()
    if checksum != DRAWS_HASH.read_text().strip():
        raise ValueError("Frozen placebo resampling draws failed SHA-256 validation")
    config = json.loads(binary)
    manifest, Y, split = fixture_inputs()
    n0 = Y.shape[0] - 1
    if (config.get("fixture_sha256") != manifest["sha256"]
        or config.get("draws_origin") != "numpy_generated_for_independent_R_execution"
        or config.get("n_donors") != n0
        or config.get("intervention_idx") != split
        or config.get("replications") != len(config.get("draws", []))
        or config.get("replications", 0) < 20):
        raise ValueError("Frozen draw metadata conflicts with donor panel")
    draws = config["draws"]
    if any(type(row) is not list or len(row) != n0
           or any(type(idx) is not int for idx in row)
           or sorted(row) != list(range(n0))
           for row in draws):
        raise ValueError("Invalid donor-bootstrap index sequence")
    return config, draws, checksum


def candidates() -> dict:
    manifest, Y, split = fixture_inputs()
    config, draws, checksum = read_draws()
    donor_panel = Y[:-1]
    effects = []
    for d in draws:
        comparison, placebo_target = donor_panel[d[:-1]], donor_panel[d[-1]]
        w, lam, _ = _sdid_weights(placebo_target, comparison, split)
        effects.append(_sdid_effect(placebo_target, comparison, split, w, lam))
    effects = np.asarray(effects)
    variance = float((len(effects)-1) / len(effects) * np.var(effects, ddof=1))
    return {"draws_sha256": checksum, "fixture_sha256": manifest["sha256"],
            "replications": len(effects), "estimates": effects.tolist(),
            "variance_candidate": variance, "statistical_certified": False}


def compare(reference: Path, *, effect_atol: float = 0.005,
            variance_atol: float = 0.005) -> dict:
    config, draws, digest = read_draws()
    ours = candidates()
    ext = json.loads(reference.read_text())
    for key, expected in {
        "method": "R_synthdid_fixed_draw_placebos",
        "reference_execution": "R_synthdid",
        "fixture_sha256": ours["fixture_sha256"],
        "draws_sha256": digest,
        "replications": ours["replications"],
        "sparsify": "none",
    "sampling_scheme": "permutation_without_replacement",
        "omega_intercept": True,
        "lambda_intercept": True,
    }.items():
        if ext.get(key) != expected:
            raise ValueError(f"Independent fixed-draw placebo metadata mismatch: {key}")
    if not isinstance(ext.get("package_version"), str) or not ext["package_version"]:
        raise ValueError("Independent reference must include executed synthdid version")
    ref_effects = np.asarray(ext.get("estimates", []), dtype=float)
    if ref_effects.shape != (len(draws),) or not np.isfinite(ref_effects).all():
        raise ValueError("Reference placebo effects must be complete and finite")
    ref_var = float(ext.get("variance_candidate", np.nan))
    expected_var = float((len(ref_effects)-1)/len(ref_effects) * np.var(ref_effects, ddof=1))
    if not np.isfinite(ref_var) or abs(ref_var - expected_var) > 1e-9:
        raise ValueError("R reference placebo variance contradicts its own effect draws")
    d_effect = float(np.max(np.abs(ref_effects - ours["estimates"])))
    d_variance = abs(ref_var - ours["variance_candidate"])
    passed = d_effect <= effect_atol and d_variance <= variance_atol
    return {"status": "numeric_candidate_pass" if passed else "numeric_candidate_fail",
            "statistical_certified": False, "reference_version": ext["package_version"],
            "reference_fixture_sha256": ours["fixture_sha256"],
            "draws_sha256": digest, "replications": len(draws),
            "max_effect_difference": d_effect, "variance_difference": d_variance,
            "tolerances": {"effects": effect_atol, "variance": variance_atol},
            "independent_r_variance": ref_var, "quasi_variance_candidate": ours["variance_candidate"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error("Authentic R synthdid fixed-draw JSON is required; parity blocked")
    result = compare(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["status"] != "numeric_candidate_pass":
        raise SystemExit("Fixed-draw R synthdid candidate parity FAILED")
    print("Numeric candidate agreement; sampling inference not certified")


if __name__ == "__main__":
    main()
