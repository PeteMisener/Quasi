"""Synthetic smoke regressions (NOT R parity) comparing quasi RDD against R rdrobust::rdrobust."""

from __future__ import annotations

import numpy as np
import pytest
import quasi


@pytest.mark.synthetic
class TestRDDSyntheticRegression:

    def test_cct_bandwidth_selection_parity(self, senate_fixture):
        df, r_golden = senate_fixture
        x_col = "margin" if "margin" in df.columns else "vote_margin_t"
        y_col = "vote" if "vote" in df.columns else "vote_share_t1"
        X = df[x_col].to_numpy()
        Y = df[y_col].to_numpy()
        h_opt, _, _ = quasi.rdd.bandwidth.select_cct_bandwidth(
            X=X, Y=Y, cutoff=0.0, kernel="triangular", order=1
        )
        np.testing.assert_allclose(h_opt, r_golden["h_optimal"], rtol=0.05)

    def test_point_estimate_and_robust_se(self, senate_fixture):
        df, r_golden = senate_fixture
        x_col = "margin" if "margin" in df.columns else "vote_margin_t"
        y_col = "vote" if "vote" in df.columns else "vote_share_t1"
        result = quasi.estimate(
            data=df,
            outcome=y_col,
            running_var=x_col,
            cutoff=0.0,
            kernel="triangular",
            order=1,
        )
        np.testing.assert_allclose(result.estimate, r_golden["estimate"], rtol=0.05)
        np.testing.assert_allclose(result.se, r_golden["se_conventional"], rtol=0.05)