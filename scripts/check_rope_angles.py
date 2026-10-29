"""Inclination of the two individual ropes (angle of AE and BD to the vertical) compared with the
longitudinal swing angle phi used in the metrics and in the MPC-SC constraint (angle of the
trolley-CoM line in the vertical longitudinal plane). In planar motion (all scenarios except S4) the
three angles coincide; in S4 the lateral and torsional motion tilts the ropes out of that plane.
Output: results/rope_angles.json"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sim import setup, scenarios as SC, plant, params as P


def job(args):
    key, name = args
    sc = SC.nominal_set()[key]
    prm, a, l = sc.prm, sc.prm[3], sc.prm[2]
    c = setup.build(name)
    c.reset()
    s = plant.initial_state(prm, xT=sc.xT0, x_com=sc.x_com0(), y_com=sc.y0, theta=sc.theta0)
    rope, phi = [], []
    for k in range(int(round(sc.T / P.TS)) + 1):
        _, _, _, _, dA, dB, _, _ = plant._kinematics(s, prm)
        rope.append([np.degrees(np.arccos(min(1.0, -d[2] / np.linalg.norm(d)))) for d in (dA, dB)])
        phi.append(np.degrees(abs(plant.swing_angle(s))))
        if k == int(round(sc.T / P.TS)):
            break
        u, _ = c(plant.measured(s))
        t = k * P.TS
        Fwx = sc.wind_x if sc.wind_t[0] <= t < sc.wind_t[1] else 0.0
        s, _, _ = plant.step_zoh(s, 1000.0 * u, Fwx, 0.0, prm, P.TS, P.H_INT)
    rope = np.array(rope)
    return key, name, dict(rope_max_deg=float(rope.max()), phi_max_deg=float(max(phi)),
                           rope_minus_phi_max_deg=float(np.max(np.abs(rope.max(axis=1) - np.array(phi)))))


if __name__ == '__main__':
    jobs = [(k, c) for k in ('S2', 'S4') for c in setup.ALL]
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(job, jobs))
    out = {f'{k}/{c}': r for k, c, r in res}
    for k, r in out.items():
        print(k, {q: round(v, 3) for q, v in r.items()})
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'rope_angles.json'), 'w'), indent=1)
