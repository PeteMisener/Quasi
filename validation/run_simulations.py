"""CLI for explicitly labeled experimental Monte Carlo evidence."""
import argparse

from validation.monte_carlo import run_simulation, write_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", nargs="+", default=["iv_strong", "iv_null", "its_ar1", "its_null"])
    p.add_argument("--reps", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", default="validation/reports/monte_carlo.json")
    args = p.parse_args()
    results = [run_simulation(name, reps=args.reps, seed=args.seed, alpha=args.alpha) for name in args.scenarios]
    from pathlib import Path

    write_report(results, Path(args.output))
    for r in results:
        print(f"{r.scenario}: completed={r.completed_reps}/{r.requested_reps}, "
              f"bias={r.bias!s}, coverage={r.coverage!s} (MCSE={r.coverage_mcse!s})")
    if any(r.failed_reps for r in results):
        raise SystemExit("Monte Carlo fit failures occurred; inspect report; do not silently discard them")


if __name__ == "__main__":
    main()
