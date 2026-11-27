"""Fail-closed method certification and strict release gates.

Checked real-data numeric references are NOT the same as independently reviewed
statistical parity, causal identification, or correct finite-sample inference.
"""
import argparse
import sys

from validation.evidence_report import PENDING_R, evaluate
from validation.reference_lab import DATASETS, read_fixture


def release_blockers():
    result = evaluate()
    blockers = []
    for row in result["references"]:
        if row["status"] != "passed_numerical_unreviewed":
            blockers.append(f"{row['id']}: {row['status']}")
        if any(c.get("status") == "blocked" for c in row.get("comparisons", [])):
            blockers.append(f"{row['id']}: unverified reference fields")
    for name in DATASETS:
        fixture = read_fixture(name)
        review = fixture["statistical_review"]
        if review.get("status") != "approved" or not review.get("reviewer"):
            blockers.append(f"{name}: independent statistical reviewer approval missing")
    if PENDING_R:
        blockers.append("Full R parity absent for: " + ", ".join(PENDING_R))
    # The original three-fixture gate is not sufficient for the expanded library.
    # Enumerate every public estimator and project-level approval requirement.
    from validation.release_readiness import audit
    readiness = audit()
    blockers.extend(blocker for method in readiness["methods"] for blocker in method["blockers"])
    blockers.extend(item["blocker"] for item in readiness["project_requirements"])
    return blockers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true", help="Require independent statistical review and all target R goldens")
    args = p.parse_args()
    report = evaluate()
    failed = [row["id"] for row in report["references"] if row["status"].startswith("failed")]
    if args.strict:
        blockers = release_blockers()
        if blockers:
            for item in blockers:
                print("BLOCKED:", item, file=sys.stderr)
            raise SystemExit(1)
    elif failed:
        raise SystemExit("Provenance/numerical fixture failure: " + ", ".join(failed))
    else:
        print("Numerical checks pass; research certification remains OPEN")


if __name__ == "__main__":
    main()
