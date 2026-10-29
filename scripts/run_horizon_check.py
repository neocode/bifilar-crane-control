"""Sensitivity of the MPC law to the prediction horizon (N = 50 versus N = 100), scenarios S1-S3."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sim import setup, controllers as C, scenarios as SC, simulate as S

_, _, Ad, Bd = setup.design_model()
ts = setup.terminal_sets()
out = {}
dmax = 0.0
for k in ('S1', 'S2', 'S3'):
    sc = SC.nominal_set()[k]
    logs = {}
    for N in (50, 100):
        mpc = C.MPC(Ad, Bd, setup.Q_MPC, setup.R_MPC, setup.RDU_MPC, N, term_set=ts['input'])
        logs[N] = S.run_nonlinear(mpc, sc)
        out[f'{k}_N{N}_term_ok'] = float(np.mean(mpc.term_ok))
    d = float(np.max(np.abs(logs[50]['u'] - logs[100]['u'])))
    out[f'{k}_max_du'] = d
    dmax = max(dmax, d)
out['max_du_N50_vs_N100'] = max(dmax, 1e-12)
print(out)
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'horizon_check.json'), 'w'), indent=1)
