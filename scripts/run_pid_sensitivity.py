"""Sensitivity of PID (optimal), PD, LQR and MPC to small rope-length errors and initial swings in the
reference case (x_T(0) = 0.5 m, otherwise nominal plant). Output: results/pid_sensitivity.json"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from concurrent.futures import ProcessPoolExecutor
from sim import setup, params as P, simulate as S, metrics as Mx


def job(a):
    name, l, phi = a
    lg = S.run_nonlinear(setup.build(name), S.Scenario('sens', xT0=0.5, phi0_deg=phi, prm=P.plant_params(l=l)))
    m = Mx.compute(lg, 0.5)
    return f'{name}/l={l}/phi0={phi}', dict(Ts=m['Ts'], overshoot=m['overshoot'])


if __name__ == '__main__':
    jobs = [(n, l, phi) for n in ('PD', 'PID (optimal)', 'LQR', 'MPC')
            for l in (5.5, 6.0, 6.5) for phi in (-1.5, 0.0, 1.5)]
    with ProcessPoolExecutor(18) as ex:
        out = dict(ex.map(job, jobs))
    for k, v in out.items():
        print(k, round(v['Ts'], 2))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'pid_sensitivity.json'), 'w'),
              indent=1)
