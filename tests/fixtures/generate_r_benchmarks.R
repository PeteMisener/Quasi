#!/usr/bin/env Rscript
# ==============================================================================
# Canonical R Ground-Truth Benchmark Generator for quasi
# Generates golden JSON benchmarks from did, rdrobust, and synthdid.
# ==============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(did)        # Callaway & Sant'Anna
  library(rdrobust)   # Cattaneo, Calonico, Titiunik
  library(synthdid)   # Arkhangelsky et al.
})

set.seed(42)

# ------------------------------------------------------------------------------
# 1. did::att_gt Benchmark (Job Training / Minimum Wage Panel: mpdta)
# ------------------------------------------------------------------------------
message("Generating did::att_gt benchmark...")
data(mpdta)

# Estimate group-time ATT using never-treated units as control
did_res <- att_gt(
  yname = "lemp",
  tname = "year",
  idname = "countyreal",
  gname = "first.treat",
  data = mpdta,
  control_group = "nevertreated",
  bstrap = TRUE,
  biters = 9999,
  cband = TRUE
)

# Aggregate overall ATT
did_agg <- aggte(did_res, type = "simple")
# Aggregate dynamic event-study
did_es <- aggte(did_res, type = "dynamic")

mpdta_golden <- list(
  package = "did",
  r_version = R.version.string,
  dataset = "mpdta",
  overall_att = as.numeric(did_agg$overall.att),
  overall_se = as.numeric(did_agg$overall.se),
  p_value = 2 * (1 - pnorm(abs(did_agg$overall.att / did_agg$overall.se))),
  event_study = list(
    egt = as.numeric(did_es$egt),
    att_egt = as.numeric(did_es$att.egt),
    se_egt = as.numeric(did_es$se.egt),
    crit_val = as.numeric(did_es$crit.val.egt)
  )
)

write_json(mpdta_golden, "tests/fixtures/mpdta_benchmark.json", pretty = TRUE, auto_unbox = TRUE)

# ------------------------------------------------------------------------------
# 2. rdrobust::rdrobust Benchmark (US Senate Elections: rdrobust_RDsenate)
# ------------------------------------------------------------------------------
message("Generating rdrobust benchmark...")
data(rdrobust_RDsenate)

X <- rdrobust_RDsenate$margin
Y <- rdrobust_RDsenate$vote

rdd_res <- rdrobust(
  Y, X,
  c = 0.0,
  kernel = "triangular",
  p = 1, # local linear
  bwselect = "mserd"
)

rdd_golden <- list(
  package = "rdrobust",
  r_version = R.version.string,
  dataset = "rdrobust_RDsenate",
  cutoff = 0.0,
  h_optimal = as.numeric(rdd_res$bws[1, 1]),
  b_pilot = as.numeric(rdd_res$bws[2, 1]),
  estimate = as.numeric(rdd_res$coef[1]),    # Conventional local linear point estimate
  se_robust = as.numeric(rdd_res$se[3]),     # Robust bias-corrected SE
  se_conventional = as.numeric(rdd_res$se[1]),
  p_value_robust = as.numeric(rdd_res$pv[3]),
  ci_robust = as.numeric(rdd_res$ci[3, ]),
  n_effective_left = as.numeric(rdd_res$N_h[1]),
  n_effective_right = as.numeric(rdd_res$N_h[2])
)

write_json(rdd_golden, "tests/fixtures/r_senate_benchmark.json", pretty = TRUE, auto_unbox = TRUE)

# ------------------------------------------------------------------------------
# 3. synthdid::synthdid_estimate Benchmark (California Prop 99)
# ------------------------------------------------------------------------------
message("Generating synthdid benchmark...")
data(california_prop99)

setup <- panel.matrices(california_prop99)
sdid_res <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
sdid_se <- sqrt(vcov(sdid_res, method = "jackknife"))

weights <- synthdid_units(sdid_res)
donor_weights <- setNames(as.numeric(weights), rownames(weights))

sdid_golden <- list(
  package = "synthdid",
  r_version = R.version.string,
  dataset = "california_prop99",
  treated_unit = "California",
  intervention_time = 1989,
  estimate = as.numeric(sdid_res),
  se_jackknife = as.numeric(sdid_se),
  p_value = 2 * (1 - pnorm(abs(as.numeric(sdid_res) / sdid_se))),
  top_donor_weights = as.list(donor_weights[donor_weights > 0.01])
)

write_json(sdid_golden, "tests/fixtures/prop99_benchmark.json", pretty = TRUE, auto_unbox = TRUE)
message("All benchmarks successfully generated.")