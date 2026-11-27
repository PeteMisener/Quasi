"""Frozen input parity contract checks do not claim R was executed locally."""
from pathlib import Path
import json

import pytest

from validation import sdid_phase10c_placebo_reference as parity


def controlled_reference():
    values = parity.candidates()
    return {
        "method": "R_synthdid_fixed_draw_placebos",
        "reference_execution": "R_synthdid",
        "fixture_sha256": values["fixture_sha256"],
        "draws_sha256": values["draws_sha256"],
        "replications": values["replications"],
        "sparsify": "none",
        "sampling_scheme": "permutation_without_replacement",
        "omega_intercept": True,
        "lambda_intercept": True,
        "package_version": "controlled-contract-test-not-external",
        "estimates": values["estimates"],
        "variance_candidate": values["variance_candidate"],
    }


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_frozen_draw_sequence_is_sha256_bound_to_panel():
    settings, draws, h = parity.read_draws()
    assert settings["replications"] == len(draws) == 40
    assert len(h) == 64
    assert settings["n_donors"] == len(draws[0]) == 12
    assert all(len(indices) == 12 for indices in draws)


def test_controlled_reference_contract_pass_is_not_statistical_certification(tmp_path):
    result = parity.compare(write(tmp_path/"test-only.json", controlled_reference()))
    assert result["status"] == "numeric_candidate_pass"
    assert not result["statistical_certified"]
    assert result["max_effect_difference"] == 0


@pytest.mark.parametrize("key,value", [
    ("method", "different_method"),
    ("reference_execution", "not_a_reference"),
    ("fixture_sha256", "not_original_data"),
    ("draws_sha256", "wrong_draws"),
    ("replications", 39),
    ("omega_intercept", False),
    ("lambda_intercept", False),
    ("sparsify", "sparse"),
    ("sampling_scheme", "with_replacement"),
    ("package_version", None),
])
def test_mismatched_reference_rejected(tmp_path, key, value):
    fixture = controlled_reference()
    fixture[key] = value
    with pytest.raises(ValueError, match="mismatch|version"):
        parity.compare(write(tmp_path/"altered.json", fixture))


def test_incomplete_draws_rejected(tmp_path):
    r = controlled_reference()
    r["estimates"] = r["estimates"][:-1]
    with pytest.raises(ValueError, match="complete"):
        parity.compare(write(tmp_path/"truncated.json", r))


def test_nonfinite_draws_rejected(tmp_path):
    r = controlled_reference()
    r["estimates"][3] = float("nan")
    with pytest.raises(ValueError, match="complete"):
        parity.compare(write(tmp_path/"not-finite.json", r))


def test_statistic_does_not_match_draw_values_rejected(tmp_path):
    r = controlled_reference()
    r["variance_candidate"] = -11
    with pytest.raises(ValueError, match="contradicts"):
        parity.compare(write(tmp_path/"badvar.json", r))


def test_effect_mismatch_is_failed_candidate_not_false_success(tmp_path):
    r = controlled_reference()
    r["estimates"] = [x+0.2 if i%2 else x-0.2 for i,x in enumerate(r["estimates"])]
    import numpy as np
    arr = np.array(r["estimates"])
    r["variance_candidate"] = (len(arr)-1)/len(arr)*np.var(arr,ddof=1)
    result = parity.compare(write(tmp_path/"disagree.json", r))
    assert result["status"] == "numeric_candidate_fail"
    assert not result["statistical_certified"]


def test_missing_real_reference_is_rejected_by_cli(tmp_path):
    import subprocess, sys
    cmd = [sys.executable, "-m", "validation.sdid_phase10c_placebo_reference",
           str(tmp_path / "absent-reference.json"), "--output", str(tmp_path / "evidence.json")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "required" in proc.stderr.lower()
    assert not (tmp_path/"evidence.json").exists()
