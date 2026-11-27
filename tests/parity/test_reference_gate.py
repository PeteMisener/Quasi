"""Parity gate against real dataset inputs and frozen third-party goldens.

Statsmodels references are independent of Quasi. Full research-grade and R
parity gates remain OPEN, particularly IV robust inference and DiD/RD/SDiD.
"""
import pytest

from validation.reference_lab import DATASETS, check_reference, compare_quasi_result, source_data


@pytest.mark.parity
@pytest.mark.parametrize("fixture_id", tuple(DATASETS))
def test_statsmodels_real_data_reference(fixture_id):
    pytest.importorskip("statsmodels", reason="validation extra missing; explicit reference workflow requires it")
    import quasi

    frozen = check_reference(fixture_id)
    method = frozen["method"]
    data = source_data(DATASETS[fixture_id][0])
    if method == "iv_2sls":
        result = quasi.TwoStageLS(
            data, outcome="invest", endogenous="value", instruments=["capital"], alpha=0.05
        ).fit()
    else:
        opts = frozen["design"]
        result = quasi.InterruptedTimeSeries(
            data, outcome="volume", time="year", intervention_date=opts["intervention_date"],
            hac_lag=opts["hac_lag"], alpha=opts["alpha"]
        ).fit()
    comparisons = compare_quasi_result(frozen, result)
    failures = [row for row in comparisons if row["status"] == "fail"]
    assert not failures, f"Reference mismatch: {failures}"
    assert any(row["status"] == "pass" for row in comparisons)
    if method == "iv_2sls":
        assert next(c for c in comparisons if c["field"] == "se")["status"] == "blocked"


@pytest.mark.parity
@pytest.mark.parametrize("method", ["staggered_did", "robust_rd", "synthetic_did"])
def test_unavailable_r_parity_is_explicitly_pending(method):
    pytest.skip(f"Authentic R {method} numerical goldens not generated or independently reviewed; no synthetic fallback")
