"""Comparative ITS simulations; not independent certification.

Run from repository root:
  PYTHONPATH=src python -m validation.its_phase9c_profile_ml --reps 160 \
     --output validation/evidence/its_phase9c_comparison.json

Scenarios use independent seeds and identical datasets across fitted methods.
All reported confidence intervals are conditional on estimated AR correlation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate(seed: int, scenario: str, n: int = 120) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    innovations=rng.normal(0,.65,n+400)
    e=np.zeros(n+400)
    if scenario=='moderate_ar1':
        a,b=.3,0.
    elif scenario=='strong_ar1':
        a,b=.85,0.
    elif scenario=='misspecified_ar2':
        a,b=.65,.25
    else:
        raise ValueError('Unsupported scenario')
    for i in range(2,len(e)):
        e[i]=a*e[i-1]+b*e[i-2]+innovations[i]
    y=2+.02*t+1.8*np.cos(2*np.pi*t/12)+.7*np.sin(2*np.pi*t/12)
    y=y+e[-n:]+1.5*(t>=60)+.012*np.maximum(0,t-60)
    return pd.DataFrame({'time':t,'y':y})


def run(*, reps: int=160, seed: int=20261009) -> dict:
    import quasi
    if type(reps) is not int or not 2<=reps<=100000 or type(seed) is not int or seed<0:
        raise ValueError('Invalid reps/seed')
    profiles=[]
    for si,scenario in enumerate(('moderate_ar1','strong_ar1','misspecified_ar2')):
        observations={m:[] for m in ('hac','ar1','ar1_mle')}
        rng=np.random.default_rng(seed+si)
        for rep in range(reps):
            data=generate(int(rng.integers(0,2**32-1)),scenario)
            for method in observations:
                row={'success':False}
                try:
                    res=quasi.InterruptedTimeSeries(
                        data,'y','time',60,error_model=method,seasonal_period=12,
                        hac_lag=3 if method=='hac' else None).fit()
                    row.update(success=True,estimate=float(res.estimate),
                               se=float(res.se),ci_lower=float(res.ci_lower),
                               ci_upper=float(res.ci_upper),
                               rho=res.inference_details.get('estimated_rho'))
                except (ValueError, ArithmeticError, RuntimeError) as exc:
                    row['exception']=type(exc).__name__
                observations[method].append(row)
        methods=[]
        for method, rows in observations.items():
            ok=[r for r in rows if r['success']]
            count=len(ok)
            covered=sum(r['ci_lower']<=1.5<=r['ci_upper'] for r in ok)
            proportion=covered/count if count else None
            mcse=float(np.sqrt(proportion*(1-proportion)/count)) if count else None
            methods.append({'error_model':method,'attempts':reps,'fits':count,
                 'fit_failures':reps-count,'coverage_95_percent':proportion,
                 'coverage_monte_carlo_se':mcse,
                 'mean_bias':float(np.mean([r['estimate']-1.5 for r in ok])) if count else None,
                 'mean_interval_width':float(np.mean([r['ci_upper']-r['ci_lower'] for r in ok])) if count else None,
                 'mean_se':float(np.mean([r['se'] for r in ok])) if count else None,
                 'mean_rho':float(np.mean([r['rho'] for r in ok])) if method!='hac' and count else None,
                 'certified':False})
        profiles.append({'scenario':scenario,'methods':methods})
    return {
      'seed':seed,'replications_per_scenario':reps,'true_level_effect':1.5,
      'confidence_level':.95,'reference_certified':False,'status':'exploratory_not_certified',
      'limitations':[
        'Only one segmented seasonal data-generating process per error structure',
        'AR1 conditional GLS covariance ignores uncertainty in estimated rho',
        'Intervals are marginal level-effect inference, not prediction intervals',
        'Monte Carlo coverage uncertainty remains substantial with limited replications',
        'Neither method is validated against additional independent small-sample reference procedures',
        'AR2 errors intentionally violate the AR1 model assumptions'
      ],
      'profiles':profiles,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=160)
    ap.add_argument('--seed',type=int,default=20261009)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=run(reps=args.reps,seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for s in result['profiles']:
        print(s['scenario'], ', '.join(f"{m['error_model']}={m['coverage_95_percent']:.3f} ({m['fits']}/{m['attempts']})" for m in s['methods']))


if __name__=='__main__':
    main()
