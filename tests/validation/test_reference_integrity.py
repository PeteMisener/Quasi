"""Integrity tests: fallbacks, fabricated fixtures, and altered data fail closed."""
from copy import deepcopy

import pytest

from validation import reference_lab


def test_reference_manifest_tampering_detected(monkeypatch):
    pytest.importorskip("statsmodels")
    name = "nile_its_95"
    authentic = reference_lab.read_fixture(name)
    real_read = reference_lab.FIXTURE_DIR
    # Replace fixture directory with one manifest with a corrupt dataset checksum.
    from tempfile import TemporaryDirectory
    from pathlib import Path
    import json

    with TemporaryDirectory() as tmp:
        changed = deepcopy(authentic)
        changed["dataset"]["sha256_canonical_csv"] = "0" * 64
        (Path(tmp) / f"{name}.json").write_text(json.dumps(changed), encoding="utf-8")
        monkeypatch.setattr(reference_lab, "FIXTURE_DIR", Path(tmp))
        with pytest.raises(ValueError, match="Dataset has changed"):
            reference_lab.read_fixture(name)
    monkeypatch.setattr(reference_lab, "FIXTURE_DIR", real_read)


def test_missing_golden_fails_without_synthetic_substitute(tmp_path, monkeypatch):
    pytest.importorskip("statsmodels")
    monkeypatch.setattr(reference_lab, "FIXTURE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="never substitute synthetic"):
        reference_lab.read_fixture("grunfeld_2sls")


def test_synthetic_manifest_rejected(tmp_path, monkeypatch):
    pytest.importorskip("statsmodels")
    import json

    frozen = reference_lab.read_fixture("grunfeld_2sls")
    frozen["reference_kind"] = "seeded_synthetic"
    (tmp_path / "grunfeld_2sls.json").write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(reference_lab, "FIXTURE_DIR", tmp_path)
    with pytest.raises(ValueError, match="Synthetic/non-independent"):
        reference_lab.read_fixture("grunfeld_2sls")


def test_unreviewed_reference_is_not_approved():
    pytest.importorskip("statsmodels")
    frozen = reference_lab.read_fixture("grunfeld_2sls")
    assert frozen["statistical_review"]["status"] == "pending"
    assert frozen["dataset"]["redistributed"] is False
    assert frozen["dataset"]["substantive_identification_claim"] is False
