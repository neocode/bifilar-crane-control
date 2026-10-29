"""Minimum and maximum rope tension over all nominal and robustness runs (taut-rope check).
Tensions are evaluated at every 0.5 ms integration node. Output: results/rope_tension_check.json"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from sim import setup, scenarios as SC, plant, params as P


@njit(cache=True)
def step_tension(s, F, Fwx, prm, T, h):
    n = int(np.round(T / h))
    tmin, tmax = 1e30, 0.0
    for _ in range(n):
        k1, t1, t2 = plant.rhs(s, F, Fwx, 0.0, prm)
        tmin = min(tmin, t1, t2)
        tmax = max(tmax, t1, t2)
        k2, _a, _b = plant.rhs(s + 0.5 * h * k1, F, Fwx, 0.0, prm)
        k3, _a, _b = plant.rhs(s + 0.5 * h * k2, F, Fwx, 0.0, prm)
        k4, _a, _b = plant.rhs(s + h * k3, F, Fwx, 0.0, prm)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return s, tmin, tmax


def job(a):
    key, sc, cname = a
    prm = sc.prm
    c = setup.build(cname)
    s = plant.initial_state(prm, xT=sc.xT0, x_com=sc.x_com0(), y_com=sc.y0, theta=sc.theta0)
    tmin, tmax = 1e30, 0.0
    for k in range(int(round(sc.T / P.TS))):
        u, _ = c(plant.measured(s))
        t = k * P.TS
        Fwx = sc.wind_x if sc.wind_t[0] <= t < sc.wind_t[1] else 0.0
        s, a1, a2 = step_tension(s, 1000.0 * u, Fwx, prm, P.TS, P.H_INT)
        tmin, tmax = min(tmin, a1), max(tmax, a2)
    return key, cname, tmin / 1000.0, tmax / 1000.0


if __name__ == '__main__':
    jobs = []
    for k, sc in list(SC.nominal_set().items()) + list(SC.robustness_set().items()):
        for c in setup.ALL:
            jobs.append((k, sc, c))
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(job, jobs))
    out = {f'{k}/{c}': dict(Tmin_kN=a, Tmax_kN=b) for k, c, a, b in res}
    mn = min(res, key=lambda r: r[2])
    mx = max(res, key=lambda r: r[3])
    out['_summary'] = dict(Tmin_kN=mn[2], Tmin_case=f'{mn[0]}/{mn[1]}', Tmax_kN=mx[3], Tmax_case=f'{mx[0]}/{mx[1]}')
    print(out['_summary'])
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'rope_tension_check.json'), 'w'), indent=1)
