"""Synthetic smoke regressions (NOT R parity) comparing quasi Staggered DiD against R did::att_gt."""

from __future__ import annotations

import numpy as np
import pytest
import quasi


@pytest.mark.synthetic
class TestDiDSyntheticRegression:

    def test_overall_att_point_estimate(self, mpdta_fixture):
        df, r_golden = mpdta_fixture
        first_treat_col = "first_treat" if "first_treat" in df.columns else "first.treat"
        model = quasi.StaggeredDiD(
            data=df,
            outcome="lemp",
            unit="countyreal",
            time="year",
            first_treated=first_treat_col,
            control_group="never_treated",
            b_reps=9999,
            seed=42,
        )
        result = model.fit()
        np.testing.assert_allclose(result.estimate, r_golden["overall_att"], rtol=0.05)

    def test_standard_error_parity(self, mpdta_fixture):
        df, r_golden = mpdta_fixture
        first_treat_col = "first_treat" if "first_treat" in df.columns else "first.treat"
        model = quasi.StaggeredDiD(
            data=df,
            outcome="lemp",
            unit="countyreal",
            time="year",
            first_treated=first_treat_col,
            control_group="never_treated",
            b_reps=9999,
            seed=42,
        )
        result = model.fit()
        np.testing.assert_allclose(result.se, r_golden["overall_se"], rtol=0.05)