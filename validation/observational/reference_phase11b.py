"""Independent real-data numerical benchmark for Quasi observational ATT.

Not a causal experiment and not an inferential certification: treatment is an
arbitrary dichotomization of a real-survey questionnaire response. This check
uses statsmodels GLM, independent closed-form ATT formulas, and a finite-
difference stacked score Jacobian rather than Quasi's fitting utilities.
Run from repository root: PYTHONPATH=src python -m validation.observational.reference_phase11b
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

HERE = Path(__file__).resolve().parent / 'fixtures'
REFERENCE = HERE / 'fair_observational_reference.json'
FEATURES = ('age', 'yrs_married', 'religious', 'educ')


def _source_csv_bytes() -> bytes:
    """Canonical snapshot from the installed Statsmodels dataset, no redistribution.

    The frozen SHA-256 in REFERENCE protects against upstream data changes;
    the independent dataset must be installed via the validation extra.
    """
    survey = sm.datasets.fair.load_pandas().data.copy()
    survey['treatment'] = (survey['rate_marriage'] >= 4).astype(int)
    return survey[['affairs', 'treatment', *FEATURES]].to_csv(
        index=False, float_format='%.14g', lineterminator='\n').encode('utf-8')


def _source_frame() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_source_csv_bytes()))


def _calc(df: pd.DataFrame) -> dict[str, float]:
    """ATT and HC0 SE from independent Statsmodels and numeric M-estimation."""
    d = df.treatment.to_numpy(dtype=float)
    y = df.affairs.to_numpy(dtype=float)
    z = np.column_stack([np.ones(len(df)), df[list(FEATURES)].to_numpy(dtype=float)])
    n, k = z.shape
    gl = sm.GLM(d, z, family=sm.families.Binomial()).fit(maxiter=500)
    gamma = np.asarray(gl.params, dtype=float)
    beta = np.asarray(sm.OLS(y[d==0],z[d==0]).fit().params)
    e = expit(z @ gamma)
    odds = (1-d)*e/(1-e)
    residual = y-z@beta
    tau = float(np.sum((d-odds)*residual)/d.sum())
    mu1 = float(np.sum(d*y)/d.sum())
    mu0 = float(np.sum(odds*y)/odds.sum())

    # Numeric differentiation is intentionally independent of Quasi's
    # analytical sandwich Jacobian.
    def stacked_se(theta: np.ndarray, f, effect) -> float:
        values = f(theta)
        h = 1e-5
        eye = np.eye(len(theta))
        jac = np.column_stack([(f(theta+h*eye[j]).mean(axis=0)-
                                f(theta-h*eye[j]).mean(axis=0))/(2*h)
                               for j in range(len(theta))])
        influence = -np.linalg.solve(jac,values.T)
        effect_if = effect(influence)
        return float(np.sqrt(np.mean((effect_if-effect_if.mean())**2)/n))

    def aipw_moments(t):
        tv = t[0]; ga = t[1:k+1]; b=t[k+1:]
        p=expit(z@ga)
        w=(1-d)*p/(1-p)
        r=y-z@b
        return np.column_stack([d*(r-tv)-w*r,(d-p)[:,None]*z,((1-d)*r)[:,None]*z])

    def ipw_moments(t):
        v1,v0=t[:2]
        p=expit(z@t[2:]); w=(1-d)*p/(1-p)
        return np.column_stack([d*(y-v1),w*(y-v0),(d-p)[:,None]*z])

    aipw_se=stacked_se(np.r_[tau,gamma,beta],aipw_moments,lambda x:x[0])
    ipw_se=stacked_se(np.r_[mu1,mu0,gamma],ipw_moments,lambda x:x[0]-x[1])
    return {'aipw_att':tau,'aipw_se_hc0':aipw_se,
            'ipw_att':mu1-mu0,'ipw_se_hc0':ipw_se,
            'n':int(n),'n_treated':int(d.sum())}


def verify_reference() -> dict:
    from quasi import AIPWATT, IPWATT, PropensityScoreMatching
    payload = json.loads(REFERENCE.read_text())
    raw = _source_csv_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != payload['data_sha256']:
        raise AssertionError('Real-data reference fixture content has changed')
    if payload['outcome'] != 'affairs' or payload['treatment'] != 'treatment' or tuple(payload['covariates']) != FEATURES:
        raise AssertionError('Reference specification mismatch')
    df = pd.read_csv(io.BytesIO(raw))
    calculated = _calc(df)
    comparisons = []
    for key, actual in calculated.items():
        expect=payload['statsmodels_reference'][key]
        if isinstance(expect,int) and isinstance(actual,int):
            ok = expect == actual
        else:
            ok = bool(np.isclose(actual, expect, rtol=1e-8, atol=1e-9))
        comparisons.append({'quantity':key,'expected':expect,'computed':actual,'passed':ok})
    a=AIPWATT(df,'affairs','treatment',list(FEATURES)).fit()
    i=IPWATT(df,'affairs','treatment',list(FEATURES)).fit()
    p=PropensityScoreMatching(df,'affairs','treatment',list(FEATURES),n_neighbors=3).fit()
    for name, actual, key in [('aipw',a.estimate,'aipw_att'),('aipw_se',a.se,'aipw_se_hc0'),
                               ('ipw',i.estimate,'ipw_att'),('ipw_se',i.se,'ipw_se_hc0')]:
        ok=bool(np.isclose(actual,calculated[key],rtol=5e-5,atol=2e-6))
        comparisons.append({'quantity':'quasi_'+name,'expected':calculated[key], 'computed':actual,'passed':ok})
    comparisons.append({'quantity':'matching_inference_absent','passed':not np.isfinite(p.se)})
    return {'passed':all(c['passed'] for c in comparisons),
            'data_sha256':digest,'reference_source':'statsmodels_glm_and_control_ols',
            'causal_claim':False,'experimental':True,'comparisons':comparisons,
            'numpy_version':np.__version__,'statsmodels_version':sm.__version__}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Independent Statsmodels real-data numerical comparison')
    parser.add_argument('--output', type=Path, default=None)
    options = parser.parse_args()
    report = verify_reference()
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)
