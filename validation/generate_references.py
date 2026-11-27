"""Generate/check immutable reference goldens from independent Statsmodels fits.

Usage: python -m validation.generate_references --check
       python -m validation.generate_references --write --acknowledge-review-required
"""
import argparse
import json
from pathlib import Path

from validation.reference_lab import DATASETS, FIXTURE_DIR, check_reference, generate_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Verify stored real-data fixtures against independent reruns")
    group.add_argument("--write", action="store_true", help="Replace pending-review goldens (explicit opt-in)")
    parser.add_argument("--acknowledge-review-required", action="store_true")
    args = parser.parse_args()
    if args.write and not args.acknowledge_review_required:
        parser.error("--write requires --acknowledge-review-required; new values do not imply statistical review")
    for name in DATASETS:
        if args.check:
            check_reference(name)
            print(f"PASS  {name}: frozen statsmodels reference reproducible")
        else:
            payload = generate_reference(name)
            FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            out = FIXTURE_DIR / f"{name}.json"
            out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            print(f"WRITE {out}: pending independent statistical review")


if __name__ == "__main__":
    main()
