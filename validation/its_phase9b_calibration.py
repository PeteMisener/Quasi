"""Exploratory ITS coverage and conditional counterfactual prediction checks.

Run: PYTHONPATH=src python -m validation.its_phase9b_calibration --reps 80 --bootstrap-reps 79 \\
  --output validation/evidence/its_phase9b.json

This is not validation or causal identification certification. Simulations intentionally
vary seasonal specification and error autocorrelation; bootstrap uses a plug-in AR1 DGP.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from quasi.errors import InsufficientSupportError


def simulate(seed: int, scenario: str, n: int = 120):
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    eps=rng.normal(0,.65,n+400)
    e=np.empty(n+400)
    e[0]=eps[0]
    e[1]=eps[1]
    if scenario=='moderate_seasonal':
        a,b=.3,0
    elif scenario=='high_seasonal':
        a,b=.85,0
    elif scenario=='misspecified_ar2_seasonal':
        a,b=.65,.25
    else:
        raise ValueError('unknown scenario')
    for k in range(2,len(e)):
        e[k]=a*e[k-1]+b*e[k-2]+eps[k]
    err=e[-n:]
    season = 1.8*np.cos(2*np.pi*t/12)+.7*np.sin(2*np.pi*t/12)
    no_treatment=2+.02*t+season+err
    treated=(t>=60).astype(float)
    y=no_treatment+1.5*treated+.012*np.maximum(0,t-60)
    return pd.DataFrame({'time':t,'y':y}),no_treatment


def run(*, reps: int=80, bootstrap_reps: int=79, seed: int=20261009):
    import quasi
    if type(reps) is not int or reps<2 or type(bootstrap_reps) is not int or bootstrap_reps<50:
        raise ValueError('Require reps>=2 and bootstrap-reps>=50')
    output=[]
    for scenario in ('moderate_seasonal','high_seasonal','misspecified_ar2_seasonal'):
        records=[]
        rng=np.random.default_rng(seed + ['moderate_seasonal','high_seasonal','misspecified_ar2_seasonal'].index(scenario))
        for k in range(reps):
            child_seed=int(rng.integers(0,2**32-1))
            data, untreated = simulate(child_seed,scenario)
            record={'rep':k,'fit':False,'bootstrap_success':False}
            try:
                result=quasi.InterruptedTimeSeries(data,'y','time',60,error_model='ar1',
                                                   seasonal_period=12).fit()
                record['fit']=True
                record['level_bias']=result.estimate-1.5
                record['model_based_hit']=bool(result.ci_lower<=1.5<=result.ci_upper)
                record['model_based_width']=result.ci_upper-result.ci_lower
                pred=result.counterfactual_prediction()
                for horizon in (1,6,12):
                    record[f'predict_h{horizon}_hit']=bool(pred.prediction_lower.iloc[horizon-1] <= untreated[59+horizon] <= pred.prediction_upper.iloc[horizon-1])
                boot=result.ar1_bootstrap_level_interval(reps=bootstrap_reps,seed=child_seed+1)
                record['bootstrap_success']=True
                record['bootstrap_hit']=bool(boot['ci_lower']<=1.5<=boot['ci_upper'])
                record['bootstrap_width']=boot['ci_upper']-boot['ci_lower']
            except (InsufficientSupportError, ValueError):
                pass
            records.append(record)
        fitted=[r for r in records if r['fit']]
        booted=[r for r in records if r['bootstrap_success']]
        def avg(rows,key):
            return float(np.mean([row[key] for row in rows])) if rows else None
        output.append({'scenario':scenario,'replications':reps,'fits':len(fitted),
             'bootstrap_successes':len(booted),
             'level_mean_bias':avg(fitted,'level_bias'),
             'model_based_coverage':avg(fitted,'model_based_hit'),
             'model_based_mean_width':avg(fitted,'model_based_width'),
             'bootstrap_coverage':avg(booted,'bootstrap_hit'),
             'bootstrap_mean_width':avg(booted,'bootstrap_width'),
             'retrospective_counterfactual_predictive_coverage':{
                 str(h):avg(fitted,f'predict_h{h}_hit') for h in (1,6,12)},
             'status':'exploratory_not_certified'})
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reps',type=int,default=80)
    ap.add_argument('--bootstrap-reps',type=int,default=79)
    ap.add_argument('--seed',type=int,default=20261009)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    results=run(reps=args.reps,bootstrap_reps=args.bootstrap_reps,seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
       'seed':args.seed, 'replications':args.reps, 'bootstrap_replications':args.bootstrap_reps,
       'confidence_level':.95, 'reference_certified':False,
       'limitations':'Exploratory small simulation; bootstrapped Gaussian AR1 DGP not robust to AR2 errors, treatment path estimated on all observed outcomes; forecast intervals ignore rho estimation and model selection',
       'profiles':results}, indent=2)+'\n')
    for x in results:
        print(x['scenario'], 'model',x['model_based_coverage'],
             'bootstrap',x['bootstrap_coverage'],
             'predict h1',x['retrospective_counterfactual_predictive_coverage']['1'],
             'fits',x['fits'],'bootstrap_success',x['bootstrap_successes'])

if __name__=='__main__': main()
