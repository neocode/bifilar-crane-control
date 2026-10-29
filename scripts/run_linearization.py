"""Validity range of the linear design model (results/linearization.json).

The same controller is simulated on the linear model and on the nonlinear plant without
friction and damping, so that the difference is caused by the geometric nonlinearity only.
Amplitude is varied through the initial trolley offset (S1-type tests) and the initial swing.
"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sim import setup, params as P, simulate as S

CASES = [('xT0', v) for v in (0.5, 1.0, 2.0, 3.0, 5.0)] + [('phi0', v) for v in (2.0, 5.0, 10.0, 15.0)]
CTRL = ['PD', 'PID (optimal)', 'LQR']


def job(a):
    kind, val, cname = a
    prm = P.plant_params(mu=0.0, cv=0.0, zeta=0.0)
    sc = S.Scenario('lin', xT0=val if kind == 'xT0' else 0.0, phi0_deg=val if kind == 'phi0' else 0.0, prm=prm)
    ln = S.run_linear(setup.build(cname), sc)
    nl = S.run_nonlinear(setup.build(cname), sc)
    xs = max(np.max(np.abs(nl['xT'])), 1e-9)
    ps = max(np.max(np.abs(nl['phi'])), 1e-12)
    return dict(kind=kind, val=val, ctrl=cname,
                phi_peak_nl=float(np.degrees(np.max(np.abs(nl['phi'])))),
                phi_peak_lin=float(np.degrees(np.max(np.abs(ln['phi'])))),
                dx_rel=float(np.max(np.abs(nl['xT'] - ln['xT'])) / xs),
                dphi_rel=float(np.max(np.abs(nl['phi'] - ln['phi'])) / ps),
                du_max=float(np.max(np.abs(nl['u'] - ln['u']))))


if __name__ == '__main__':
    jobs = [(k, v, c) for (k, v) in CASES for c in CTRL]
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(job, jobs))
    for r in res:
        print(f"{r['kind']}={r['val']:5.1f} {r['ctrl']:14s} peak swing {r['phi_peak_nl']:6.2f} deg (lin {r['phi_peak_lin']:6.2f})"
              f"  max dx/xmax={100*r['dx_rel']:6.2f}%  max dphi/phimax={100*r['dphi_rel']:6.2f}%")
    json.dump(res, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'linearization.json'), 'w'), indent=1)
