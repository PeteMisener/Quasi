# Independent execution of frozen donor-bootstrap placebo draws in R synthdid.
# Usage: Rscript validation/r/phase10c_sdid_fixed_draw_placebos.R OUT.json
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Expected output JSON path")
if (!requireNamespace("synthdid", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Real R synthdid and jsonlite dependencies are required")
}
panel.file <- "validation/fixtures/sdid_phase10b_panel.csv"
manifest <- jsonlite::fromJSON("validation/fixtures/sdid_phase10b_panel.json", simplifyVector = TRUE)
draws.file <- "validation/fixtures/sdid_phase10c_draws.json"
config <- jsonlite::fromJSON(draws.file, simplifyVector = FALSE)
sha <- function(path) trimws(strsplit(system2("sha256sum", path, stdout = TRUE), " ")[[1]][1])
if (sha(panel.file) != manifest$sha256) stop("Panel fixture hash mismatch")
if (sha(draws.file) != readLines("validation/fixtures/sdid_phase10c_draws.sha256")[[1]]) stop("Draw hash mismatch")
if (config$fixture_sha256 != manifest$sha256 ||
    config$sampling_scheme != "permutation_without_replacement") stop("Draws do not match canonical scheme or panel")
panel <- read.csv(panel.file, stringsAsFactors = FALSE)
donors <- sort(setdiff(unique(panel$unit), manifest$treated_unit))
periods <- sort(unique(panel$time))
N0 <- length(donors)
T0 <- match(manifest$intervention_time, periods) - 1L
if (T0 < 2L || T0 >= length(periods)) stop("Invalid split")
Y <- matrix(NA_real_, N0, length(periods))
for (i in seq_along(donors)) {
  rows <- panel[panel$unit == donors[i], ]
  Y[i,] <- rows$y[match(periods, rows$time)]
}
if (anyNA(Y)) stop("Donor outcomes missing")
if (config$n_donors != N0 || config$intervention_idx != T0 || length(config$draws) != config$replications) {
  stop("Draw metadata mismatch")
}
effects <- vapply(config$draws, function(indices) {
  idx <- as.integer(unlist(indices)) + 1L
  if (length(idx) != N0 || !identical(sort(idx), as.integer(seq_len(N0)))) stop("Draw must be a complete donor permutation")
  matrix.draw <- Y[idx,,drop=FALSE]
  as.numeric(synthdid::synthdid_estimate(matrix.draw, N0=N0-1L, T0=T0,
                sparsify=NULL, omega.intercept=TRUE, lambda.intercept=TRUE))
}, numeric(1))
B <- length(effects)
variance <- (B-1)/B * stats::var(effects)
reference <- list(method="R_synthdid_fixed_draw_placebos", reference_execution="R_synthdid",
                  package_version=as.character(utils::packageVersion("synthdid")),
                  fixture_sha256=manifest$sha256, draws_sha256=sha(draws.file),
                  replications=B, sparsify="none", sampling_scheme="permutation_without_replacement", omega_intercept=TRUE,
                  lambda_intercept=TRUE, estimates=unname(as.numeric(effects)),
                  variance_candidate=as.numeric(variance))
jsonlite::write_json(reference, args[1], auto_unbox=TRUE, pretty=TRUE, digits=15)
