"""Dedicated single-process timing of the MPC QP solvers and terminal-set statistics
(results/qp_timing.json). Run on an otherwise idle machine."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sim import setup, scenarios as SC, simulate as S, params as P

out = {}
allt = {'MPC': [], 'MPC-SC': []}
for name in ('MPC', 'MPC-SC'):
    setup.build(name)                       # warm-up (terminal sets, imports)
    for k, sc in SC.nominal_set().items():
        c = setup.build(name)
        S.run_nonlinear(c, sc)
        st = np.array(c.solve_times) * 1e3
        allt[name].append(st)
        ok = np.array(c.term_ok, bool)
        bad = np.where(~ok)[0]
        t_ok = 0.0 if len(bad) == 0 else (bad[-1] + 1) * P.TS
        out[f'{name}/{k}'] = dict(mean_ms=float(st.mean()), p99_ms=float(np.percentile(st, 99)),
                                  max_ms=float(st.max()), term_ok_frac=float(ok.mean()),
                                  term_ok_from_s=float(t_ok), iters_mean=float(np.mean(c.iters)))
        print(name, k, out[f'{name}/{k}'])
for name in allt:
    st = np.concatenate(allt[name])
    out[name] = dict(mean_ms=float(st.mean()), p99_ms=float(np.percentile(st, 99)), max_ms=float(st.max()))
    print(name, out[name])
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'qp_timing.json'), 'w'), indent=1)
