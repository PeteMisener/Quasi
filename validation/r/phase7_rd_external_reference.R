#!/usr/bin/env Rscript
# Independently executed R parity reference. Validation of provenance and
# conventions is not a substitute for statistical approval.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop('Expected input data CSV and output CSV')
for (pkg in c('rdrobust','digest')) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop(paste(pkg,'is required'))
}
expected_hash <- 'cd6f007fb4d3e0fa01a97e8d8bee8c994516d725d4227ec1d8807099b2814854'
actual_hash <- digest::digest(file=args[[1]], algo='sha256')
if (!identical(actual_hash, expected_hash)) stop('Fixture SHA-256 mismatch; refusing parity output')
raw <- read.csv(args[[1]])
if (!identical(names(raw), c('x','y','d')) || nrow(raw)!=1800 ||
    any(!is.finite(as.matrix(raw))) || any(!(raw$d %in% c(0,1))) ||
    !any(raw$x < 0) || !any(raw$x >= 0)) stop('Invalid frozen fixture')
output <- list()
for (design in c('sharp','fuzzy')) {
  opts <- list(y=raw$y, x=raw$x, c=0, p=1, q=2,
               h=1.15, b=1.35, kernel='tri', vce='hc0')
  if (design=='fuzzy') opts$fuzzy <- raw$d
  fit <- do.call(rdrobust::rdrobust, opts)
  if (length(fit$coef) < 2 || length(fit$se) < 3) {
    stop('Unexpected rdrobust result layout: inspect reference API')
  }
  output[[length(output)+1]] <- data.frame(
    design=design, method=c('conventional','bias_corrected'),
    estimate=c(as.numeric(fit$coef[1]),as.numeric(fit$coef[2])),
    se=c(as.numeric(fit$se[1]),as.numeric(fit$se[3])),
    reference_package='rdrobust',
    reference_version=as.character(utils::packageVersion('rdrobust')),
    fixture_sha256=actual_hash, h=1.15, b=1.35, p=1, q=2,
    vce='hc0', kernel='tri', cutoff=0,
    generator='validation/r/phase7_rd_external_reference.R',
    stringsAsFactors=FALSE)
}
result <- do.call(rbind,output)
if (any(!is.finite(result$estimate)) || any(!is.finite(result$se)) ||
    any(result$se <= 0)) stop('Reference returned invalid estimates or SEs')
write.csv(result,args[[2]],row.names=FALSE)
cat('Produced',args[[2]],'using R rdrobust',as.character(utils::packageVersion('rdrobust')),'\n')
