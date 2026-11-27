"""Coverage for reporting, visualization, power, AI, and prep/Polars modules."""
from __future__ import annotations

import json
import urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import quasi
from quasi.ai.adapters import GeminiAdapter, MockAdapter
from quasi.ai.condenser import condense_metadata_payload
from quasi.errors import InsufficientSupportError, InvalidDesignError
from quasi.prep.audit import audit_panel
from quasi.prep.extractor import extract_column_arrays


def _sample_results():
    rng = np.random.default_rng(401)
    # IV
    z = rng.normal(size=120)
    x = 1.2 * z + rng.normal(size=120)
    df_iv = pd.DataFrame({"y": 1.5 * x + rng.normal(size=120), "x": x, "z": z})
    r_iv = quasi.estimate(df_iv, design=quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",)))

    # SCM (point-only)
    rows = []
    for u in ("t", "c1", "c2", "c3", "c4"):
        for p in range(10):
            rows.append({"u": u, "p": p, "y": 5.0 + 0.3 * p + (2.0 if u == "t" and p >= 6 else 0.0) + rng.normal(0, 0.1)})
    df_sc = pd.DataFrame(rows)
    r_scm = quasi.estimate(df_sc, design=quasi.SyntheticDesign(
        outcome="y", unit="u", time="p", treated_unit="t", intervention_time=6, estimator="scm", n_placebos=2
    ))
    r_sdid = quasi.estimate(df_sc, design=quasi.SyntheticDesign(
        outcome="y", unit="u", time="p", treated_unit="t", intervention_time=6, estimator="sdid", n_placebos=2
    ))

    # PSM (point-only NaN SE)
    x1 = rng.normal(size=160)
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * x1)))
    df_obs = pd.DataFrame({"y": 2.0 * d + x1 + rng.normal(size=160), "d": d, "x1": x1})
    r_psm = quasi.estimate(df_obs, design=quasi.ObservationalDesign(
        outcome="y", treatment="d", covariates=("x1",), estimator="psm"
    ))
    return {"IV": r_iv, "SCM": r_scm, "SDiD": r_sdid, "PSM": r_psm}


def test_compare_models_all_renderers_and_point_only_safety(tmp_path):
    results = _sample_results()
    for style in ("aer", "nature"):
        tbl = quasi.compare_models(results, title="Cross-Method Check", style=style)
        latex = tbl.to_latex(str(tmp_path / f"t_{style}.tex"))
        md = tbl.to_markdown(str(tmp_path / f"t_{style}.md"))
        typst = tbl.to_typst(str(tmp_path / f"t_{style}.typ"))
        html_out = tbl.to_html(str(tmp_path / f"t_{style}.html"))
        assert "\\begin{tabular}" in latex
        assert "| **Panel A: Causal Estimates** |" in md
        # Verify fixed Typst syntax: balanced empty header cell "[]" (not unclosed "[")
        assert "    [], [*(1)" in typst
        assert "<table" in html_out
        assert "Diagnostic Status" in md

    # Every result (including point-only SCM, SDiD, PSM) renders HTML & explain without crashing
    for res in results.values():
        card = res._repr_html_()
        assert "<div" in card and "nan" not in card.lower().split("n/a")[0]
        for aud in ("academic", "stakeholder", "student"):
            txt = res.explain(audience=aud, engine="rule_based")
            assert isinstance(txt, str) and len(txt) > 20


def test_all_visualizations_render_cleanly():
    results = _sample_results()
    for res in results.values():
        fig, _ = res.plot()
        plt.close(fig)

    # synth_plots honors passed ax_traj / ax_gap
    from quasi.visualization.synth_plots import render_synth_figure
    fig_ext, (ax1, ax2) = plt.subplots(2, 1)
    fig_ret, (r1, r2) = render_synth_figure(
        ["0", "1", "2", "3", "4"], np.arange(5, dtype=float), np.arange(5, dtype=float), 3,
        ax_traj=ax1, ax_gap=ax2,
    )
    assert fig_ret is fig_ext and r1 is ax1 and r2 is ax2
    plt.close(fig_ext)

    # ITS plot
    t = np.arange(30)
    df_its = pd.DataFrame({"t": t, "y": 5.0 + 0.1 * t + 2.0 * (t >= 15)})
    r_its = quasi.InterruptedTimeSeries(df_its, "y", "t", 15).fit()
    fig_its, _ = r_its.plot()
    plt.close(fig_its)


def test_power_ar1_vs_compound_symmetry_and_infeasible_guards():
    # Under multi-period positive serial correlation (T0=4, T1=4, rho=0.7), AR(1) variance
    # exceeds compound-symmetry variance because positive autocorrelation reduces effective T.
    p_ar1 = quasi.did_mde(40, 8, 4, serial_correlation=0.7, correlation_structure="ar1")
    p_cs = quasi.did_mde(40, 8, 4, serial_correlation=0.7, correlation_structure="compound_symmetry")
    assert p_ar1.mde > p_cs.mde
    fig_p, _ = p_ar1.plot()
    plt.close(fig_p)

    # Custom empirical covariance matrix matches AR(1) exact matrix
    idx = np.arange(8)
    ar1_cov = 0.7 ** np.abs(idx[:, None] - idx[None, :])
    p_emp = quasi.did_mde(40, 8, 4, error_cov=ar1_cov)
    assert p_emp.mde == pytest.approx(p_ar1.mde, rel=1e-10)

    # Sample size solvers raise InsufficientSupportError rather than silently returning max cap
    with pytest.raises(InsufficientSupportError):
        quasi.did_sample_size(target_mde=0.001, n_periods=6, pre_periods=3, max_clusters=20)
    with pytest.raises(InsufficientSupportError):
        quasi.rdd_sample_size(target_mde=0.001, max_n=200)

    # RDD RBC power has larger MDE than conventional
    r_conv = quasi.rdd_mde(1000, inference="conventional")
    r_rbc = quasi.rdd_mde(1000, inference="bias_corrected")
    assert r_rbc.mde > r_conv.mde

    # Synth feasibility
    sf = quasi.synth_feasibility(n_donors=15, n_pre_periods=20)
    assert not sf.is_permutation_viable
    assert "INFEASIBLE" in sf.status

    # Simulation-based power engine
    spec = quasi.IVDesign(outcome="y", endogenous="x", instruments=("z",))
    def dgp(rng):
        z = rng.normal(size=120)
        x = 1.0 * z + rng.normal(size=120)
        y = 1.5 * x + rng.normal(size=120)
        return pd.DataFrame({"y": y, "x": x, "z": z})
    sim = quasi.simulate_power(spec, dgp, n_sims=20, seed=7)
    assert sim.n_succeeded == 20 and sim.empirical_power > 0.8


def test_ai_condenser_point_only_and_gemini_header_security(monkeypatch):
    results = _sample_results()
    # Condenser handles None/NaN point-only metadata without TypeError
    condensed = condense_metadata_payload(results["SCM"].to_metadata_dict())
    assert condensed["primary_metric"]["se"] is None
    assert condensed["primary_metric"]["p_value"] is None
    assert condensed["primary_metric"]["ci"] == [None, None]
    assert len(json.dumps(condensed)) < 4096

    # GeminiAdapter MUST send API key via x-goog-api-key header, NEVER in URL query string
    captured = {}
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}).encode()
    def fake_urlopen(req, timeout=45):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        return FakeResp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    gem = GeminiAdapter(api_key="super-secret-key")
    resp = gem.generate("sys", "usr")
    assert resp.content == "ok"
    assert "super-secret-key" not in captured["url"]
    assert captured["headers"].get("X-goog-api-key") == "super-secret-key"

    # MockAdapter & ask()
    quasi.ai.configure(provider="mock")
    assert "DesignSpec" in quasi.ask("How do I run DiD?")
    assert MockAdapter().generate("DIAGNOSTIC", "check").provider == "mock"


def test_prep_audit_panel_and_polars_like_eager_and_lazy_extractors():
    # Eager Polars-like protocol (has .lazy(), .select(), .to_arrow(), NO .collect())
    class EagerFrame:
        def __init__(self, tbl): self._t = tbl
        def lazy(self): return LazyFrame(self._t)
        def select(self, cols): return EagerFrame(self._t.select(cols))
        def to_arrow(self): return self._t

    class LazyFrame:
        def __init__(self, tbl): self._t = tbl
        def select(self, cols): return LazyFrame(self._t.select(cols))
        def collect(self): return EagerFrame(self._t)

    table = pa.table({"y": [1.0, 2.0, 3.0], "x": [4.0, 5.0, 6.0]})
    out_eager = extract_column_arrays(EagerFrame(table), {"outcome": "y", "covariates": ["x"]})
    out_lazy = extract_column_arrays(LazyFrame(table), {"outcome": "y", "covariates": ["x"]})
    assert np.array_equal(out_eager["outcome"], [1.0, 2.0, 3.0])
    assert np.array_equal(out_lazy["outcome"], [1.0, 2.0, 3.0])

    # audit_panel checks duplicates and removes obsolete misleading banner claims
    df = pd.DataFrame({
        "u": [1, 1, 2, 2, 2],
        "t": [0, 1, 0, 1, 1],
        "d": [0, 1, 0, 0, 0],
    })
    aud = audit_panel(df, "u", "t", "d")
    assert aud.duplicate_key_count == 1
    assert not aud.is_viable
    banner = aud.format_banner()
    assert "Wild Bootstrap activated" not in banner
    assert "Observations dropped or interpolated" not in banner
