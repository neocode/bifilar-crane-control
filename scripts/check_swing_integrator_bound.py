"""Role of the anti-windup bound (24) of the swing-angle integrator in PID (optimal).

Without saturation, the swing integrator is equivalent to an absolute load-velocity feedback with gain
-K_ia/g (Sect. 3.4.4). The script repeats the regulation test of the tuning (linear design model,
x_T(0) = 0.5 m) with and without the bound |K_ia xi_a| <= F_max, and counts the samples at which the
bound is active in the reference case S1 on the nonlinear plant. Output: results/swing_integrator_bound.json
"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sim import params as P, setup, controllers as C, scenarios as SC, simulate as S, metrics as Mx
from sim.linear import c2d

KEYS = ['KpT', 'KiT', 'KdT', 'Kpa', 'Kia', 'Kda']


def regulation_test(g, swing_bound=True, x0=0.5, T=P.T_END, Ts=P.TS, Fmax=P.F_MAX, l=P.L_ROPE):
    """Same discrete closed loop and anti-windup as sim/tuning.py:pid_sim_linear."""
    A, B, _, _ = P.linear_model()
    Ad, Bd = c2d(A, B, Ts)
    KpT, KiT, KdT, Kpa, Kia, Kda = g
    x, xiT, xia, clipped, xT, ka = np.array([x0, 0.0, 0.0, 0.0]), 0.0, 0.0, 0, [], []
    for _ in range(int(round(T / Ts)) + 1):
        xT.append(x[0])
        eT, al = -x[0], x[2] / l
        u_un = KpT * eT + KiT * xiT - KdT * x[1] + Kpa * al + Kia * xia + Kda * x[3] / l
        u = min(max(u_un, -Fmax), Fmax)
        sat = abs(u_un) > Fmax + P.SAT_TOL
        if KiT != 0.0 and not (sat and np.sign(KiT * eT) == np.sign(u_un)):
            b = Fmax / abs(KiT)
            xiT = min(max(xiT + Ts * eT, -b), b)
        if Kia != 0.0 and not (sat and np.sign(Kia * al) == np.sign(u_un)):
            b = Fmax / abs(Kia) if swing_bound else np.inf
            new = xia + Ts * al
            clipped += abs(new) > b
            xia = min(max(new, -b), b)
        ka.append(Kia * xia)
        x = Ad @ x + Bd[:, 0] * u
    xT = np.array(xT)
    return dict(overshoot=float(max(0.0, -xT.min()) / x0 * 100), clipped_steps=int(clipped),
                max_Kia_xia=float(np.max(np.abs(ka))))


class Recording(C.PID):
    def reset(self):
        super().reset()
        self.at_bound = 0

    def __call__(self, z, x_ref=0.0):
        out = super().__call__(z, x_ref)
        self.at_bound += abs(abs(self.Kia * self.xia) - self.Fmax) < 1e-9
        return out


if __name__ == '__main__':
    gd = setup.load_gains()['PID (optimal)']['gains']
    g = [gd[k] for k in KEYS]
    c = -g[4] / P.G
    eq = list(g)
    eq[2], eq[4], eq[5] = g[2] - c, 0.0, g[5] + c * P.L_ROPE
    out = dict(velocity_gain=c, equivalent_gains=dict(zip(KEYS, eq)),
               with_bound=regulation_test(g, True), without_bound=regulation_test(g, False),
               equivalent=regulation_test(eq, True))
    ctrl = Recording(g, name='PID (optimal)')
    log = S.run_nonlinear(ctrl, SC.nominal_set()['S1'])
    out['S1_samples_at_bound'] = int(ctrl.at_bound)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'swing_integrator_bound.json'), 'w'),
              indent=1)
