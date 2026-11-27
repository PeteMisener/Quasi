"""Show that a green numerical comparison cannot unlock research certification."""
import pytest

from validation import gate
from validation.evidence_report import evaluate


def test_current_release_gate_fails_closed():
    pytest.importorskip("statsmodels")
    issues = gate.release_blockers()
    assert any("reviewer approval missing" in issue for issue in issues)
    assert any("Full R parity absent" in issue for issue in issues)
    assert any("unverified reference fields" in issue for issue in issues)


def test_evidence_dashboard_labels_no_causal_certification():
    pytest.importorskip("statsmodels")
    report = evaluate()
    assert report["validation_release_approved"] is False
    assert any(row["status"] == "pending" for row in report["references"])
    assert all(row["status"] != "approved" for row in report["references"])
