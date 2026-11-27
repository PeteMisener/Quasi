"""Strict candidate comparison against *executed* external R reference files.

This consumes producer output from ``validation/r/phase6_did_external_reference.R``.
It never generates R benchmarks itself. A numerical pass never means validated
causal identification, coverage or independent reviewer approval.

    python -m validation.compare_phase6_r output/r_phase6 --report report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from quasi.did.conditional import traditional_dr_panel
from quasi.did.staggered import StaggeredDiD


_REFERENCE_NAMES = [f'mpdta_{control}_{base}_a0' for control in ('nevertreated', 'notyettreated')
                    for base in ('varying', 'universal')]


def _validate_producer(root: Path) -> dict[str, str]:
    versions = root / 'versions.txt'
    if not versions.exists():
        raise FileNotFoundError('No executed R producer outputs: versions.txt missing')
    metadata = dict(line.split('=', 1) for line in versions.read_text().splitlines() if '=' in line)
    if not {'R', 'did', 'DRDID'} <= set(metadata):
        raise ValueError('Incomplete R producer package-version manifest')
    hashes = pd.read_csv(root / 'checksums.csv')
    if hashes.empty or set(hashes.columns) != {'filename', 'md5'}:
        raise ValueError('R-produced checksums.csv is missing or malformed')
    if hashes.filename.duplicated().any() or hashes.filename.str.contains(r'[/\\]').any():
        raise ValueError('Duplicate or invalid checksum inventory filenames')
    for row in hashes.itertuples(index=False):
        path = root / row.filename
        if not path.is_file() or hashlib.md5(path.read_bytes()).hexdigest() != row.md5:
            raise ValueError(f'Corrupted R reference input/output: {row.filename}')
    required = {f'{n}.csv' for n in _REFERENCE_NAMES} | {
        'mpdta_input.csv', 'drdid_input.csv', 'drdid_traditional_ref.csv', 'drdid_traditional_if.csv'
    }
    if not required <= set(hashes.filename):
        raise ValueError('R reference manifest is missing required comparison files')
    return metadata


def compare(root: Path, *, tolerance: float = 2e-7, raise_on_mismatch: bool = True) -> dict:
    """Compare cell points and DRDID point/IF/SE; no reference certification."""
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError('tolerance must be finite positive')
    root = Path(root)
    metadata = _validate_producer(root)
    source = pd.read_csv(root / 'mpdta_input.csv')
    required_columns = {'lemp', 'countyreal', 'year', 'first.treat'}
    if not required_columns <= set(source):
        raise ValueError('R input mpdta columns incomplete')
    results = {}
    for control in ('nevertreated', 'notyettreated'):
        for base in ('varying', 'universal'):
            name = f'mpdta_{control}_{base}_a0'
            reference = pd.read_csv(root / f'{name}.csv')
            if not {'g', 't', 'att', 'se'} <= set(reference):
                raise ValueError(f'{name}: expected g,t,att,se columns')
            if reference[['g','t']].duplicated().any():
                raise ValueError(f'{name}: duplicate cohort-time benchmark keys')
            actual = StaggeredDiD(
                source, outcome='lemp', unit='countyreal', time='year',
                first_treated='first.treat',
                control_group='never_treated' if control == 'nevertreated' else 'not_yet_treated',
                base_period=base, b_reps=19,
            ).fit().group_time_frame().rename(columns={'cohort':'g','time':'t', 'estimate':'quasi_att','se':'quasi_se'})
            # R's normalized zero reference cells do not have a stochastic
            # Quasi counterpart. Preserve their count rather than treating
            # them as mismatched estimated effects.
            reference = reference[['g','t','att','se']].dropna(subset=['att'])
            aligned = reference.merge(actual[['g','t','quasi_att','quasi_se']], on=['g','t'], how='outer', indicator=True)
            normalized = (aligned['_merge'] == 'left_only') & (aligned['att'].fillna(np.inf).abs() <= tolerance)
            mismatches = aligned[(aligned['_merge'] != 'both') & ~normalized]
            common = aligned[aligned['_merge']=='both'].copy()
            error = float(np.max(abs(common.att-common.quasi_att))) if len(common) else float('inf')
            se_delta = abs(common.se-common.quasi_se)
            results[name] = {
                'reference_count':len(reference), 'quasi_count':len(actual), 'matched_count':len(common),
                'normalized_R_placeholders':int(normalized.sum()),
                'unsupported_or_unmatched':int(len(mismatches)),
                'unmatched_examples':mismatches[['g','t','_merge']].head(8).astype(str).to_dict('records'),
                'point_max_abs_difference':error,
                'se_max_abs_difference_informational':float(se_delta.max()) if len(common) else None,
                'point_parity_candidate':len(common)>0 and len(mismatches)==0 and error <= tolerance,
                'se_parity_certified':False,
            }

    drdata = pd.read_csv(root / 'drdid_input.csv')
    rref = pd.read_csv(root / 'drdid_traditional_ref.csv')
    rif = pd.read_csv(root / 'drdid_traditional_if.csv')
    columns = ['age','educ','black','married','nodegree','hisp','re74']
    if not set(columns + ['re78','re75','experimental']) <= set(drdata):
        raise ValueError('DRDID input columns are incomplete')
    cell = traditional_dr_panel((drdata.re78 - drdata.re75).to_numpy(),
                                drdata.experimental.to_numpy(), drdata[columns].to_numpy())
    if len(rref)!=1 or set(['att','se']) - set(rref) or len(rif)!=len(cell.influence):
        raise ValueError('Malformed DRDID reference output')
    if len(rif.columns)!=1:
        raise ValueError('DRDID influence reference should be one column')
    oracle_if = rif.iloc[:,0].to_numpy(dtype=float)
    if not np.isfinite(oracle_if).all():
        raise ValueError('Nonfinite DRDID influence reference')
    point_delta = float(abs(cell.att - float(rref.att.iloc[0])))
    se_est = float(np.linalg.norm(cell.influence) / len(cell.influence))
    rse = float(rref.se.iloc[0])
    se_relative = abs(se_est-rse)/max(1e-12, abs(rse))
    if_max = float(np.max(np.abs(cell.influence-oracle_if)))
    results['traditional_dr_pair'] = {
        'point_max_abs_difference':point_delta, 'influence_max_abs_difference':if_max,
        'se_relative_difference':float(se_relative), 'quasi_se':se_est, 'reference_se':rse,
        'point_parity_candidate':point_delta<=tolerance,
        'inference_parity_candidate':if_max<=1e-4 and se_relative<=.01,
        'independent_inference_reviewed':False,
    }
    passed = all(r['point_parity_candidate'] for r in results.values()) and results['traditional_dr_pair']['inference_parity_candidate']
    output = {'candidate_parity_passed':bool(passed),
              'research_certified':False,
              'qualification':'External numerical comparison only; not coverage or methodological signoff',
              'R_versions':metadata, 'results':results}
    if not passed and raise_on_mismatch:
        raise AssertionError(f'R candidate parity/inference mismatch: {json.dumps(output, default=str)}')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--tolerance', type=float, default=2e-7)
    args = parser.parse_args()
    # Write a detailed report even if parity fails, so CI exposes divergence.
    result = compare(args.directory, tolerance=args.tolerance, raise_on_mismatch=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps(result, indent=2, default=str))
    if not result['candidate_parity_passed']:
        raise SystemExit('Candidate parity mismatch (see report); statistical gate remains BLOCKED')

if __name__ == '__main__':
    main()
