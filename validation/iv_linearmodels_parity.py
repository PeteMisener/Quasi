"""Fail-closed external comparison to linearmodels, on a deterministic input contract.

Run with: python -m validation.iv_linearmodels_parity --output evidence.json
This is an independent *numerical* parity check, never a causal certification.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd


def run() -> dict:
    try:
        from linearmodels.iv import IV2SLS, IVGMM
    except ImportError as exc:
        raise RuntimeError('The independent linearmodels package is required; no parity was performed') from exc
    from quasi.iv.tsls import TwoStageLS
    fixture=Path(__file__).parent/'fixtures'/'iv_phase8_synthetic.csv'
    expected=(Path(__file__).parent/'fixtures'/'iv_phase8_synthetic.sha256').read_text().strip()
    digest=hashlib.sha256(fixture.read_bytes()).hexdigest()
    if digest!=expected:
        raise RuntimeError('Frozen IV fixture SHA-256 mismatch: refusing parity')
    df=pd.read_csv(fixture)
    exog=pd.DataFrame({'intercept':np.ones(len(df)),'w':df.w})
    results=[]
    for cov in ('HC0','CR0'):
        k=dict(cluster='group',covariance='CR0') if cov=='CR0' else dict(covariance='HC0')
        q=TwoStageLS(df,'y','x',['z0','z1'],covariates=['w'],**k).fit()
        ref=IV2SLS(df.y,exog,df[['x']],df[['z0','z1']]).fit(
            cov_type='clustered' if cov=='CR0' else 'robust',
            debiased=False, **({'clusters':df['group']} if cov=='CR0' else {}))
        got=float(ref.params['x'])
        se=float(ref.std_errors['x'])
        if not np.isclose(q.estimate,got,rtol=1e-8,atol=1e-9) or not np.isclose(q.se,se,rtol=1e-7,atol=1e-8):
            raise AssertionError(f'Independent IV mismatch for {cov}: Quasi ({q.estimate},{q.se}), reference ({got},{se})')
        results.append({'covariance':cov,'quasi_estimate':q.estimate,'reference_estimate':got,
                        'quasi_se':q.se,'reference_se':se,'numerical_agreement':True})
    # Separate two-step, uncentered heteroskedastic GMM J comparison.
    # This is NOT a 2SLS Sargan statistic and is NOT a validity certificate.
    reference_gmm=IVGMM(df.y,exog,df[['x']],df[['z0','z1']],
                        weight_type='robust',center=False,debiased=False).fit(
                            iter_limit=2, cov_type='robust',debiased=False)
    quasi_gmm=TwoStageLS(df,'y','x',['z0','z1'],covariates=['w'],covariance='HC0').fit().gmm_overidentification_test()
    j_ref=float(reference_gmm.j_stat.stat)
    gmm_beta_ref=float(reference_gmm.params['x'])
    if not np.isclose(quasi_gmm.statistic,j_ref,rtol=1e-6,atol=1e-7) or not np.isclose(
        quasi_gmm.gmm_coefficients[1],gmm_beta_ref,rtol=1e-7,atol=1e-8):
        raise AssertionError(f'IV-GMM J mismatch: Quasi J={quasi_gmm.statistic}, '
                             f'reference J={j_ref}, Quasi beta={quasi_gmm.gmm_coefficients[1]}, '
                             f'reference beta={gmm_beta_ref}')
    results.append({'method':'auxiliary_two_step_IVGMM_HC0_uncentered',
                    'quasi_j':quasi_gmm.statistic, 'reference_j':j_ref,
                    'quasi_gmm_beta':quasi_gmm.gmm_coefficients[1],
                    'reference_gmm_beta':gmm_beta_ref,
                    'degrees_of_freedom':quasi_gmm.degrees_of_freedom,
                    'numerical_agreement':True})
    return {'dataset_sha256':digest,'reference_package':'linearmodels',
            'reference_version':importlib.metadata.version('linearmodels'),
            'validation_tier':'numerical_only_unreviewed','experimental':True,'results':results}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Fail before writing the success artifact when missing or disagreeing.
    payload=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    main()
