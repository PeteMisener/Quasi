#!/usr/bin/env Rscript
# Reproducible R *candidate* reference generator. Never marks outputs approved.
# Execute with: Rscript validation/reference_r/generate_pending_goldens.R /tmp/r_pending_goldens.json
# Prerequisites: did, rdrobust, synthdid, jsonlite, digest; pin versions externally using renv.lock.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop("Usage: Rscript .../generate_pending_goldens.R OUTPUT.json")
required <- c("did", "rdrobust", "synthdid", "jsonlite", "digest")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing)) stop(paste("Missing R reference packages:", paste(missing, collapse=", ")))
set.seed(20261009)

# Store canonical text snapshot fingerprints, not serialized R objects.
dataset_hash <- function(df) {
  path <- tempfile(fileext=".csv")
  on.exit(unlink(path))
  write.csv(df, file=path, row.names=FALSE, na="NA")
  digest::digest(file=path, algo="sha256", file=TRUE)
}

package_versions <- as.list(vapply(required, function(name) as.character(packageVersion(name)), character(1)))

# 1) DiD authentic research dataset supplied inside did.
data("mpdta", package="did", envir=environment())
did_df <- get("mpdta", envir=environment())
did_fit <- did::att_gt(
  yname="lemp", tname="year", idname="countyreal", gname="first.treat",
  data=did_df, xformla=~1, control_group="nevertreated",
  est_method="reg", bstrap=FALSE, panel=TRUE, anticipation=0L,
  base_period="varying", allow_unbalanced_panel=FALSE
)
did_agg <- did::aggte(did_fit, type="simple")
did_output <- list(
  rows=nrow(did_df), columns=colnames(did_df), sha256_export_csv=dataset_hash(did_df),
  all_group_period_att=as.numeric(did_fit$att), all_group_period_group=as.numeric(did_fit$group),
  all_group_period_time=as.numeric(did_fit$t),
  aggregate_att=as.numeric(did_agg$overall.att), aggregate_se=as.numeric(did_agg$overall.se),
  inference="asymptotic analytic, no bootstrap", data_source="did::mpdta"
)

# 2) Senate regression discontinuity reference (includes missing outcomes).
data("rdrobust_RDsenate", package="rdrobust", envir=environment())
rd_df <- get("rdrobust_RDsenate", envir=environment())
valid <- complete.cases(rd_df[, c("vote", "margin")])
rd_fit <- rdrobust::rdrobust(y=rd_df$vote[valid], x=rd_df$margin[valid], c=0, p=1)
rd_output <- list(
  rows_original=nrow(rd_df), rows_complete=sum(valid), sha256_export_csv=dataset_hash(rd_df),
  estimate=unname(as.numeric(rd_fit$Estimate)), conventional_se=unname(as.numeric(rd_fit$se[1, 1])),
  all_ci=unname(as.matrix(rd_fit$ci)), bandwidths=unname(as.matrix(rd_fit$bws)),
  data_source="rdrobust::rdrobust_RDsenate", inference="rdrobust default"
)

# 3) California Prop 99 SDiD reference; placebo SE is fragile for a single treated unit.
data("california_prop99", package="synthdid", envir=environment())
sdid_df <- get("california_prop99", envir=environment())
setup <- synthdid::panel.matrices(sdid_df)
sdid_fit <- synthdid::synthdid_estimate(setup$Y, setup$N0, setup$T0)
sdid_se <- sqrt(as.numeric(synthdid::vcov(sdid_fit, method="placebo")))
sdid_output <- list(
  rows=nrow(sdid_df), columns=colnames(sdid_df), sha256_export_csv=dataset_hash(sdid_df),
  n_controls=setup$N0, n_pre=setup$T0,
  estimate=as.numeric(sdid_fit), placebo_se=sdid_se,
  data_source="synthdid::california_prop99",
  caveat="single treated unit; placebo variance uncertainty not independently verified"
)

manifest <- list(
  status="pending_external_review", schema_version=1L, seed=20261009L,
  r_version=R.version.string, package_versions=package_versions,
  not_approved=TRUE, datasets_not_redistributed=TRUE,
  designs=list(did=did_output, rd=rd_output, sdid=sdid_output)
)
jsonlite::write_json(manifest, path=args[1], pretty=TRUE, auto_unbox=TRUE, digits=16, na="null")
cat("Wrote candidate reference results ONLY (statistical review pending):", args[1], "\n")
