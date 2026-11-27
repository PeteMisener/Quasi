"""Synthetic smoke regressions; historical fake R targets are NOT parity evidence."""

from __future__ import annotations

import numpy as np
import pytest
import quasi


@pytest.mark.synthetic
class TestSynthSyntheticRegression:

    def test_sdid_treatment_effect_point_estimate(self, prop99_fixture):
        df, _unverified_golden = prop99_fixture
        unit_col = "State" if "State" in df.columns else "state"
        time_col = "Year" if "Year" in df.columns else "year"
        y_col = "PacksPerCapita" if "PacksPerCapita" in df.columns else "packs_per_capita"
        def estimate():
            return quasi.estimate(data=df, outcome=y_col, unit=unit_col,
                                  time=time_col, treated_unit="California",
                                  intervention_time=1989, n_placebos=10, seed=42)
        a, b = estimate(), estimate()
        assert np.isfinite(a.estimate)
        assert a.estimate == pytest.approx(b.estimate, abs=1e-9)
        assert a.inference_details["reference_certified"] is False

    def test_unverified_jackknife_not_advertised_as_se(self, prop99_fixture):
        df, _ = prop99_fixture
        unit_col = "State" if "State" in df else "state"
        time_col = "Year" if "Year" in df else "year"
        outcome_col = "PacksPerCapita" if "PacksPerCapita" in df else "packs_per_capita"
        r = quasi.estimate(data=df, outcome=outcome_col, unit=unit_col,
                           time=time_col, treated_unit="California",
                           intervention_time=1989, n_placebos=0)
        assert np.isnan(r.se)
        assert r.to_metadata_dict()["primary_metric"]["se"] is None
        assert r.inference_details["method"] == "not_available"
