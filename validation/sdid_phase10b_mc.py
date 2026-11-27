"""Point-estimator-only exploratory SDiD simulation; no CI coverage claims.

Run: python -m validation.sdid_phase10b_mc --reps 200 --output results.json
"""
from __future__ import annotations
import argparse
import json

import numpy as np
from quasi.synth.sdid import _sdid_weights, _sdid_effect


def run(*, reps=200, seed=20261009, n_donors=10, t_pre=9, t_post=5,
        truth=2.0) -> dict:
    if reps<1 or n_donors<3 or t_pre<3 or t_post<1:
        raise ValueError('Invalid simulation settings')
    results={k: [] for k in ('clean_donors','post_shocked_donor')}
    failures={k:0 for k in results}
    rng=np.random.default_rng(seed)
    T=t_pre+t_post
    for _ in range(reps):
        t=np.arange(T)
        factors=rng.normal(size=(2,T))
        loadings=rng.normal(size=(n_donors,2))
        donors=3 + .2*t + loadings@factors + rng.normal(scale=.3,size=(n_donors,T))
        true_w=np.r_[0.45, rng.dirichlet(np.ones(n_donors-1))*0.55]
        treated= 2 + true_w@donors + truth*(t>=t_pre) + rng.normal(scale=.25,size=T)
        contaminated=donors.copy()
        contaminated[0,t_pre:]+=5
        for scenario,controls in [('clean_donors',donors),('post_shocked_donor',contaminated)]:
            try:
                w,lam,_=_sdid_weights(treated,controls,t_pre)
                results[scenario].append(_sdid_effect(treated,controls,t_pre,w,lam))
            except Exception:
                failures[scenario]+=1
    return {
        'status':'synthetic_exploratory_not_reference_certified',
        'seed':seed,'reps_requested':reps,'n_donors':n_donors,
        'pre_periods':t_pre,'post_periods':t_post,'target_att':truth,
        'scenarios':{
            key:{'completed':len(values),'failures':failures[key],
                 'mean_estimate':float(np.mean(values)) if values else None,
                 'bias':float(np.mean(values)-truth) if values else None,
                 'rmse':float(np.sqrt(np.mean((np.array(values)-truth)**2))) if values else None}
            for key,values in results.items()},
        'limitations':[
            'Synthetic point-estimator study only, not repeated-sampling interval coverage',
            'The treated outcome is constructed partly from the donor convex hull',
            'An unobserved donor-only post-treatment shock can bias SDiD',
            'No external R synthdid run has been executed in this environment',
        ],
    }


def main():
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--reps',type=int,default=200)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    output=run(reps=args.reps)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output['scenarios'],indent=2))


if __name__=='__main__':
    main()
