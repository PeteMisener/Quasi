"""DGP and simulation framework correctness, not estimator research approval."""

import numpy as np
import pytest

from validation.dgp import SCENARIOS, generate
from validation.monte_carlo import run_simulation, wilson


@pytest.mark.parametrize("name", tuple(SCENARIOS))
def test_dgp_is_seeded_and_has_finite_known_truth(name):
    a = generate(name, 123)
    b = generate(name, 123)
    assert a.data.equals(b.data)
    assert np.isfinite(a.truth)
    assert not a.data.empty


def test_weak_instrument_is_weaker():
    strong = generate("iv_strong", 11)
    weak = generate("iv_weak", 11)
    corr_strong = strong.data["x"].corr(strong.data["z"])
    corr_weak = weak.data["x"].corr(weak.data["z"])
    assert abs(corr_strong) > abs(corr_weak)


def test_did_anticipation_is_identification_stress_case():
    parallel = generate("did_parallel", 11)
    anticipation = generate("did_anticipation", 11)
    assert not parallel.data.equals(anticipation.data)
    assert anticipation.truth == parallel.truth


def test_simulations_are_deterministic_and_account_for_all_reps():
    a = run_simulation("iv_strong", reps=6, seed=148)
    b = run_simulation("iv_strong", reps=6, seed=148)
    assert a == b
    assert a.completed_reps + a.failed_reps == a.requested_reps
    assert a.coverage_wilson_95 is not None
    assert 0 <= a.coverage_wilson_95[0] <= a.coverage_wilson_95[1] <= 1


def test_null_simulations_report_rejection_rate():
    res = run_simulation("its_null", reps=5, seed=18)
    assert res.truth == 0
    assert res.rejection_rate is not None
    assert res.bias_mcse is not None


def test_wilson_bounds_and_invalid_reps():
    assert wilson(0, 20)[0] >= 0
    assert wilson(20, 20)[1] <= 1
    with pytest.raises(ValueError):
        wilson(2, 1)
    with pytest.raises(ValueError):
        run_simulation("iv_strong", reps=1)
    with pytest.raises(ValueError, match="fixture-only"):
        run_simulation("synthetic_shared_trend", reps=5)
