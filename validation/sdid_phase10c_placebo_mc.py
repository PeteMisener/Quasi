"""Exploratory scaling check of reference-style placebo spread, never inference approval."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from quasi.synth.sdid import SyntheticDiD
from validation.sdid_phase10c_audit_mc import generate


def run(reps: int=30,placebos: int=30,seed: int=20261110):
    effects=[]; spreads=[]
    for i in range(reps):
        frame,truth=generate(seed+i,contaminated=False)
        r=SyntheticDiD(frame,'y','unit','time','treated',7,treatment='treated',n_placebos=0).fit()
        candidate=r.experimental_placebo_variance(replications=placebos,seed=seed+i+1)
        effects.append(float(r.estimate));spreads.append(candidate['spread_candidate'])
    values=np.asarray(effects);s=np.asarray(spreads)
    return {'experiment':'uncalibrated_placebo_variance_scaling','replications':reps,
            'draws_per_replication':placebos,'seed':seed,'true_att':truth,
            'estimator_empirical_standard_deviation':float(np.std(values,ddof=1)),
            'candidate_placebo_mean_spread':float(np.mean(s)),
            'candidate_to_empirical_spread_ratio':float(np.mean(s)/np.std(values,ddof=1)),
            'sampling_inference_certified':False,'ci_coverage_measured':False,
            'interpretation':'Different variance scales in this DGP do not establish CI validity; independent inference remains blocked.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reps',type=int,default=30)
    parser.add_argument('--draws',type=int,default=30)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=run(a.reps,a.draws)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
