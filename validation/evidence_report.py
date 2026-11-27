"""Generate reference evidence dashboard; numerical checks do not confer approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.reference_lab import DATASETS, check_reference, compare_quasi_result, source_data

# Declared external R benchmarks not yet available: no fabricated values.
PENDING_R = {
    "did_mpdta_r": "R did::mpdta ungenerated/unreviewed",
    "rdd_senate_r": "R rdrobust::rdrobust_RDsenate ungenerated/unreviewed",
    "sdid_prop99_r": "R synthdid::california_prop99 ungenerated/unreviewed",
}


def evaluate() -> dict:
    import quasi

    rows = []
    for fixture_id in DATASETS:
        try:
            fixture = check_reference(fixture_id)
            data = source_data(DATASETS[fixture_id][0])
            if fixture["method"] == "iv_2sls":
                fit = quasi.TwoStageLS(data, "invest", "value", ["capital"]).fit()
            else:
                config = fixture["design"]
                fit = quasi.InterruptedTimeSeries(data, "volume", "year", config["intervention_date"],
                                                  hac_lag=config["hac_lag"], alpha=config["alpha"]).fit()
            comparisons = compare_quasi_result(fixture, fit)
            failed = any(c["status"] == "fail" for c in comparisons)
            status = "failed_numerical" if failed else "passed_numerical_unreviewed"
            rows.append({
                "id": fixture_id, "method": fixture["method"], "dataset": fixture["dataset"]["loader"],
                "data_sha256": fixture["dataset"]["sha256_canonical_csv"],
                "reference_version": fixture["reference"]["version"],
                "review": fixture["statistical_review"]["status"],
                "status": status, "comparisons": comparisons,
            })
        except (FileNotFoundError, ValueError, AssertionError, ImportError) as exc:
            rows.append({"id": fixture_id, "status": "failed_provenance", "reason": str(exc)})
    for name, reason in PENDING_R.items():
        rows.append({"id": name, "status": "pending", "reason": reason})
    return {"schema_version": 1, "validation_release_approved": False, "references": rows}


def write_report(path: Path):
    result = evaluate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = ["# Quasi independent-reference status", "",
                "**EXPERIMENTAL: NO STABLE ESTIMATOR APPROVAL.**", "",
                "| Fixture | State | Reference | Blocked fields |",
                "|---|---|---|---|"]
    for row in result["references"]:
        status = row["status"]
        blocked = ", ".join(c["field"] for c in row.get("comparisons", []) if c["status"] == "blocked")
        markdown.append(f"| {row['id']} | {status} | {row.get('reference_version', 'R pending')} | {blocked or '—'} |")
    path.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if any(row["status"].startswith("failed") for row in result["references"]):
        raise RuntimeError("Reference evidence contains numerical or provenance failures")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="validation/reports/reference_status.json")
    args = p.parse_args()
    result = write_report(Path(args.output))
    for row in result["references"]:
        print(f"{row['id']}: {row['status']}")


if __name__ == "__main__":
    main()
