"""Paired donor contamination stress test; no statistical certification claims."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from quasi.errors import InvalidDesignError
from quasi.synth.sdid import SyntheticDiD


def generate(seed: int, *, contaminated: bool, n_donors: int=6, periods: int=12, split: int=7):
    rng=np.random.default_rng(seed)
    t=np.arange(periods)
    common=.15*t + .35*np.sin(t*.5)
    donors=np.array([common+idx*.18 +rng.normal(0,.08,periods) for idx in range(n_donors)])
    clean_treated=3 + .62*donors[0]+.38*donors[1] + rng.normal(0,.07,periods)
    treated=clean_treated+2.0*(t>=split)
    exposed=np.zeros_like(donors,dtype=int)
    if contaminated:
        donors[0,split:]+=4.0
        exposed[0,split:]=1
    rows=[('treated',int(i),float(y),int(i>=split)) for i,y in enumerate(treated)]
    for j in range(n_donors):
        rows.extend((f'd{j}',int(i),float(y),int(exposed[j,i])) for i,y in enumerate(donors[j]))
    return pd.DataFrame(rows,columns=['unit','time','y','treated']), 2.0


def run(reps=80,seed=20261009):
    if not isinstance(reps,int) or reps<5:raise ValueError('reps>=5')
    clean=[];contaminated=[]; blocked=0
    for i in range(reps):
        for shock,store in [(False,clean),(True,contaminated)]:
            frame,target=generate(seed+i,contaminated=shock)
            fit=SyntheticDiD(frame,'y','unit','time','treated',7,
                             treatment=None,n_placebos=0).fit()
            store.append(float(fit.estimate-target))
            if shock:
                try:
                    SyntheticDiD(frame,'y','unit','time','treated',7,
                                 treatment='treated',n_placebos=0).fit()
                except InvalidDesignError:
                    blocked+=1
                else:
                    raise AssertionError('Donor exposure should have blocked fitting')
            else:
                checked=SyntheticDiD(frame,'y','unit','time','treated',7,
                                     treatment='treated',n_placebos=0).fit()
                assert abs(checked.estimate-fit.estimate)<1e-10
    def stats(values):
        arr=np.asarray(values)
        return {'mean_bias':float(np.mean(arr)),'rmse':float(np.sqrt(np.mean(arr**2))),
                'replications':len(arr)}
    return {'experiment':'donor_contamination_observable_treatment_stress',
            'seed':seed,'truth':2.0,'clean':stats(clean),
            'unverified_contaminated':stats(contaminated),
            'contaminated_panels_blocked_with_treatment_audit':blocked,
            'r_parity_executed':False,'sampling_inference_certified':False,
            'interpretation':'Intentionally contaminated donors; audit validates observed treatment only, not spillovers.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=80)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(reps=a.reps);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
