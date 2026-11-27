# Quasi v1.0 method-by-method release readiness

**RELEASE BLOCKED. Every estimator listed below is experimental.**

Methods audited: **10**. Outstanding requirements: **37**.

| Estimator | Status | Reference needed | Principal limitations |
|---|---|---|---|
| `staggered_did` | Experimental, blocked | R did / DRDID | Conditional design and treatment-timing restrictions |
| `sharp_rdd` | Experimental, blocked | R/Python rdrobust | Provisional density diagnostics and heuristic legacy bandwidth |
| `fuzzy_rdd` | Experimental, blocked | R/Python rdrobust | Inference and first-stage uncertainty remain experimental |
| `iv_2sls` | Experimental, blocked | Python linearmodels / Statsmodels | Single-endogenous design and unavailable multiway clustering |
| `its` | Experimental, blocked | Statsmodels OLS/GLS/SARIMAX | Observed undercoverage under strong AR(1) and misspecified AR(2) |
| `scm` | Experimental, blocked | R Synth / validated SCM reference | Outcome-only donor matching and unsupported primary SE |
| `sdid` | Experimental, blocked | R synthdid | Uncertified primary SE/CI intentionally unavailable |
| `aipw_att` | Experimental, blocked | Independent DR ATT implementation | Observed undercoverage in some correct-nuisance simulations |
| `ipw_att` | Experimental, blocked | Independent weighted ATT implementation | Cross-fitted IPW primary uncertainty unavailable |
| `psm` | Experimental, blocked | Independent k-NN matching reference | Primary SE/CI intentionally unavailable |

## Project-wide blockers

- Owner approval and distribution-name registry ownership
- Independent software/data license and provenance review
- Passing fresh-dependency Python/OS CI and reference workflows
- Maintainer-approved public API and breaking-change migration
- Named independent statistical sign-off for each stable estimator
- Security contact, scanning, release documentation and user guidance
- Method-specific test coverage, adverse-case and simulation gates

## Evidence interpretation

Local numerical equality and synthetic simulations establish neither statistical coverage nor identification. References must be executed and reviewed independently. These requirements remain open until signed off through an accountable release process.
