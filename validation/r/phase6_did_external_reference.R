# External-reference producer ONLY. Do not call or import Quasi in this script.
# Execute in a separately maintained R environment with `did` and `DRDID`.
# Usage: Rscript validation/r/phase6_did_external_reference.R output/r_phase6
# Record actual package versions and datasets; never overwrite reviewed fixtures
# without a human-reviewed change and exact reproduction check.
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) == 1) args[1] else stop('Provide output directory')
if (!requireNamespace('did', quietly = TRUE)) stop('Install did first')
if (!requireNamespace('DRDID', quietly = TRUE)) stop('Install DRDID first')
dir.create(out, recursive = TRUE, showWarnings = FALSE)
writeLines(c(paste0('R=', getRversion()), paste0('did=', packageVersion('did')),
             paste0('DRDID=', packageVersion('DRDID'))), file.path(out, 'versions.txt'))

data('mpdta', package = 'did')
write.csv(mpdta, file.path(out, 'mpdta_input.csv'), row.names = FALSE)
for (control in c('nevertreated', 'notyettreated')) {
  for (base in c('varying', 'universal')) {
    for (a in 0:1) {
      fit <- did::att_gt(yname = 'lemp', tname = 'year', idname = 'countyreal',
                          gname = 'first.treat', data = mpdta, panel = TRUE,
                          xformla = ~1, control_group = control, anticipation = a,
                          est_method = 'reg', base_period = base,
                          bstrap = FALSE, cband = FALSE)
      write.csv(data.frame(g = fit$group, t = fit$t, att = fit$att, se = fit$se),
                file.path(out, sprintf('mpdta_%s_%s_a%d.csv', control, base, a)), row.names = FALSE)
    }
  }
}

# A traditional DRDID 2x2 panel comparator with observed data, not
# an upgraded/improved DR variant. Quasi's conditional `dr` implements
# the *traditional* estimator; do not compare it to `did::att_gt` default.
data('nsw', package = 'DRDID')
pair <- subset(nsw, treated == 0 | sample == 2)
vars <- c('re78', 're75', 'experimental', 'age', 'educ', 'black', 'married',
          'nodegree', 'hisp', 're74')
pair <- pair[complete.cases(pair[, vars]), vars]
# Stratified deterministic sample: avoid accidental all-control prefixes.
set.seed(20261009)
case <- which(pair$experimental == 1)
control <- which(pair$experimental == 0)
if (length(case) < 150L || length(control) < 150L) stop('NSW reference has insufficient treated/controls')
pair <- pair[c(sample(case, 150L), sample(control, 250L)), , drop = FALSE]
pair <- pair[order(pair$experimental, pair$age), , drop = FALSE]
if (length(unique(pair$experimental)) != 2L) stop('Invalid DRDID comparator sample')
X <- as.matrix(cbind('(Intercept)' = 1, pair[, c('age','educ','black','married','nodegree','hisp','re74')]))
fit <- DRDID::drdid_panel(y1 = pair$re78, y0 = pair$re75,
                          D = pair$experimental, covariates = X, inffunc = TRUE, trim.level = 0.999)
write.csv(pair, file.path(out, 'drdid_input.csv'), row.names = FALSE)
write.csv(data.frame(att = fit$ATT, se = fit$se),
          file.path(out, 'drdid_traditional_ref.csv'), row.names = FALSE)
write.csv(data.frame(influence = fit$att.inf.func),
          file.path(out, 'drdid_traditional_if.csv'), row.names = FALSE)

# R-produced artifact inventory for accidental corruption detection. This is
# NOT cryptographic attestation of independent execution or reviewer approval.
files <- list.files(out, pattern = '[.]csv$', full.names = TRUE)
hashes <- tools::md5sum(files)
write.csv(data.frame(filename=basename(files), md5=unname(hashes)),
          file.path(out, 'checksums.csv'), row.names=FALSE)
