"""Dedicated single-process timing of the MPC QP solvers and terminal-set statistics
(results/qp_timing.json). Run on an otherwise idle machine. Two times are recorded per sampling
step: the QP time (set-up of the constraint data and solution) and the complete control computation
(right-hand side of the QP, QP, terminal-set check)."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sim import setup, scenarios as SC, simulate as S, params as P
import time


class Timed:
    """Records the wall-clock time of every complete controller call."""

    def __init__(self, ctrl):
        self.c, self.total = ctrl, []

    def reset(self):
        self.c.reset()
        self.total = []

    def __call__(self, z, x_ref=0.0):
        t0 = time.perf_counter()
        out = self.c(z, x_ref)
        self.total.append(time.perf_counter() - t0)
        return out


out = {}
allt = {'MPC': [], 'MPC-SC': []}
alltot = {'MPC': [], 'MPC-SC': []}
for name in ('MPC', 'MPC-SC'):
    setup.build(name)                       # warm-up (terminal sets, imports)
    for k, sc in SC.nominal_set().items():
        tc = Timed(setup.build(name))
        S.run_nonlinear(tc, sc)
        c = tc.c
        st = np.array(c.solve_times) * 1e3
        tot = np.array(tc.total) * 1e3
        allt[name].append(st)
        alltot[name].append(tot)
        ok = np.array(c.term_ok, bool)
        bad = np.where(~ok)[0]
        t_ok = 0.0 if len(bad) == 0 else (bad[-1] + 1) * P.TS
        out[f'{name}/{k}'] = dict(mean_ms=float(st.mean()), p99_ms=float(np.percentile(st, 99)),
                                  max_ms=float(st.max()), total_mean_ms=float(tot.mean()),
                                  total_p99_ms=float(np.percentile(tot, 99)), total_max_ms=float(tot.max()),
                                  n_over_Ts=int(np.sum(tot > 1e3 * P.TS)),
                                  t_last_over_Ts=float(np.nonzero(tot > 1e3 * P.TS)[0].max() * P.TS) if np.any(tot > 1e3 * P.TS) else 0.0,
                                  iters_max=int(np.max(c.iters)),
                                  term_ok_frac=float(ok.mean()),
                                  term_ok_from_s=float(t_ok), iters_mean=float(np.mean(c.iters)))
        print(name, k, out[f'{name}/{k}'])
for name in allt:
    st, tot = np.concatenate(allt[name]), np.concatenate(alltot[name])
    out[name] = dict(mean_ms=float(st.mean()), p99_ms=float(np.percentile(st, 99)), max_ms=float(st.max()),
                     total_mean_ms=float(tot.mean()), total_p99_ms=float(np.percentile(tot, 99)),
                     total_max_ms=float(tot.max()), n_over_Ts=int(np.sum(tot > 1e3 * P.TS)))
    print(name, out[name])
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'qp_timing.json'), 'w'), indent=1)
