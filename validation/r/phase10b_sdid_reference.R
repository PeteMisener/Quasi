# Generate an *independent* synthdid reference file from a frozen input CSV.
# Usage: Rscript validation/r/phase10b_sdid_reference.R OUT.json
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Expected one output JSON path")
if (!requireNamespace("synthdid", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Install R packages synthdid and jsonlite before executing this reference")
}
input <- "validation/fixtures/sdid_phase10b_panel.csv"
metadata <- jsonlite::fromJSON("validation/fixtures/sdid_phase10b_panel.json")
actual.hash <- trimws(strsplit(system2("sha256sum", input, stdout = TRUE), " ")[[1]][1])
if (!identical(actual.hash, metadata$sha256)) stop("Input hash differs from frozen benchmark manifest")
panel <- read.csv(input, stringsAsFactors = FALSE)
units <- sort(setdiff(unique(panel$unit), metadata$treated_unit))
periods <- sort(unique(panel$time))
if (nrow(panel) != (length(units)+1)*length(periods) || anyDuplicated(panel[c('unit','time')])) stop("Invalid panel")
Y <- matrix(NA_real_, nrow=length(units)+1, ncol=length(periods))
for (i in seq_along(c(units, metadata$treated_unit))) {
  u <- c(units, metadata$treated_unit)[i]
  rows <- panel[panel$unit == u,]
  Y[i,] <- rows$y[match(periods, rows$time)]
}
N0 <- length(units)
T0 <- match(metadata$intervention_time, periods) - 1L
if (anyNA(Y) || T0 < 2 || T0 >= length(periods)) stop("Invalid Y or preperiod")
# Disable reference's optional second sparsification pass; use the default
# first-difference noise, eta.omega, eta.lambda, both intercepts.
fit <- synthdid::synthdid_estimate(Y, N0=N0, T0=T0,
                                   sparsify=NULL, omega.intercept=TRUE,
                                   lambda.intercept=TRUE)
weights <- attr(fit,'weights')
reference <- list(method="R_synthdid_estimate", reference_execution="R_synthdid",
  package_version=as.character(utils::packageVersion('synthdid')),
  fixture_sha256=metadata$sha256, design=metadata$design,
  sparsify="none", omega_intercept=TRUE, lambda_intercept=TRUE,
  estimate=as.numeric(fit), omega=unname(as.numeric(weights$omega)),
  lambda=unname(as.numeric(weights$lambda)))
jsonlite::write_json(reference, path=args[1], auto_unbox=TRUE, pretty=TRUE, digits=15)
