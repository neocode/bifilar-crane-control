"""Sensitivity of the reference case S1 to the smoothing velocity v_s of the tanh-regularized Coulomb
friction (results/friction_vs.json). The regularization has no stiction, so the trolley can creep under
net forces below the Coulomb force; final errors below the settling band therefore depend on v_s."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from concurrent.futures import ProcessPoolExecutor
from sim import setup, params as P, simulate as S, metrics as Mx

CTRL = ['PD', 'PID (optimal)', 'LQR', 'MPC']
VS = [0.002, 0.0005, 0.0002]


def job(a):
    name, vs = a
    m = Mx.compute(S.run_nonlinear(setup.build(name), S.Scenario('S1', xT0=0.5, prm=P.plant_params(vs=vs))), 0.5)
    return name, vs, {k: float(m[k]) for k in ('Ts', 'Ts2', 'overshoot', 'phi_peak', 'e_final_mm')}


if __name__ == '__main__':
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(job, [(c, v) for c in CTRL for v in VS]))
    out = {}
    for name, vs, m in res:
        out.setdefault(f'{1000 * vs:g}', {})[name] = m
        print(name, vs, {k: round(v, 3) for k, v in m.items()})
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'friction_vs.json'), 'w'), indent=1)
