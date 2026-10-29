"""Integration-accuracy check of the evaluation plant (results/integration_check.txt).

Reference: SciPy solve_ivp (DOP853, rtol = 1e-11, atol = 1e-12) over every sampling interval
with the same zero-order-hold input. Compared: fixed-step RK4 with h = 0.5 ms (used in all runs)
and h = 0.25 ms.
"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.integrate import solve_ivp
from sim import params as P, plant, setup, scenarios as SC

out = []


def run(ctrl, sc, mode, h=P.H_INT):
    prm = sc.prm
    s = plant.initial_state(prm, xT=sc.xT0, x_com=sc.x_com0(), y_com=sc.y0, theta=sc.theta0)
    ctrl.reset()
    n = int(round(sc.T / P.TS))
    X = np.zeros((n + 1, plant.NSTATE))
    for k in range(n + 1):
        X[k] = s
        if k == n:
            break
        u, _ = ctrl(plant.measured(s))
        t = k * P.TS
        Fwx = sc.wind_x if sc.wind_t[0] <= t < sc.wind_t[1] else 0.0
        if mode == 'rk4':
            s, _, _ = plant.step_zoh(s, 1000.0 * u, Fwx, 0.0, prm, P.TS, h)
        else:
            sol = solve_ivp(lambda tt, ss: plant.rhs(ss, 1000.0 * u, Fwx, 0.0, prm)[0], (0, P.TS), s,
                            method='DOP853', rtol=1e-11, atol=1e-12)
            s = sol.y[:, -1]
    return X


for sk, cname in [('S1', 'PID (optimal)'), ('S2', 'LQR'), ('S4', 'MPC')]:
    sc = SC.nominal_set()[sk]
    ref = run(setup.build(cname), sc, 'ivp')
    for h in (0.0005, 0.00025):
        X = run(setup.build(cname), sc, 'rk4', h)
        dxT = np.max(np.abs(X[:, 0] - ref[:, 0]))
        dphi = np.degrees(np.max(np.abs(np.arctan2(X[:, 1] - X[:, 0], -X[:, 3])
                                        - np.arctan2(ref[:, 1] - ref[:, 0], -ref[:, 3]))))
        dth = np.degrees(np.max(np.abs(X[:, 4] - ref[:, 4])))
        s = (f'{sk} {cname:14s} RK4 h={h*1e3:.2f} ms vs DOP853: max|dx_T|={dxT:.2e} m, '
             f'max|dphi|={dphi:.2e} deg, max|dtheta|={dth:.2e} deg')
        print(s)
        out.append(s)
open(os.path.join(os.path.dirname(__file__), '..', 'results', 'integration_check.txt'), 'w').write('\n'.join(out) + '\n')
