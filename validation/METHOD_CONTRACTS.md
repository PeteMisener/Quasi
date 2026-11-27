# Numerical validation method contracts (preliminary; independent approval pending)

**Never interpret point parity as proof of design identification or valid variance.** These are frozen engineering checks to detect regressions, not declarations of methodological quality.

| Method | External reference | Committed input | Active comparisons | Blocked comparisons | Target research acceptance |
|---|---|---|---|---|---|
| IV 2SLS | Statsmodels sandbox IV2SLS 0.14.6 | Grunfeld full 220-row panel via loader + SHA256 | 2SLS point (atol=1e-9, rtol=1e-8), sample N exact, first-stage F (atol=1e-7, rtol=1e-8) | HC0-like Quasi SE vs homoskedastic Statsmodels SE cannot be equated | Full HC0/HC1 and cluster covariance independently verified; rank/weak-IV simulations |
| ITS segmented | Statsmodels OLS/HAC 0.14.6 | Nile 100-year dataset via loader + SHA256 | Level effect, HAC SE, p-value, 90% and 95% CIs, slope effect, slope SE, N (numeric atol=1e-7, rtol=1e-8) | None in *numerical* interface; causal identification and finite-sample calibration unreviewed | Autocorrelation/lag robustness, false-positive and CI coverage calibrated |
| Staggered DiD | R did, pending package/version lock | did::mpdta (authentic source, no local golden) | NONE | Entire group-time ATT/SE/aggregation/bootstrap signature | Cohort eligibility, influence functions, simultaneous bands, coverage independently signed off |
| Sharp/fuzzy RD | R rdrobust, pending package/version lock | rdrobust::rdrobust_RDsenate | NONE | Robust bias-corrected CI, bandwidth, density, fuzzy first stage | Precisely comparable data cutoff, mass points, variance and bandwidth selection |
| Synthetic DiD | R synthdid, pending package/version lock | synthdid::california_prop99 | NONE | Donor/time weights, regularization, placebo/jackknife inference | Individual method weights, N0/T0, placebo finite-sample uncertainty |
| Matching/AIPW | pending independent benchmark | pending | NONE | Full inference and matching estimand | Real matched effect, overlap, nuisance/crossfit correction |

The reference dataset producer must not import Quasi; stored goldens must have checksums, backend versions, row/column signatures and reproducing scripts. Every active field has a recorded absolute and relative tolerance. Solver and stochastic methods must use convergence and uncertainty acceptance standards, not an arbitrary hard 0.25% numerical tolerance. Required reviewer signatures and source/legal audit are separate gate items.

The current `validation/gate.py --strict` intentionally fails. This file documents incomplete scopes so no later feature expansion can redefine the release gate without an explicit decision.
