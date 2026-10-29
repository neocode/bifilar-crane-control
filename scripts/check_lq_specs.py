"""LQR and MPC evaluated with the tuning index and specifications of the PD/PID designs
(results/lq_specs.json). The LQR and MPC weights were chosen by hand, not by the specification-based
tuning; this script evaluates them in the same two linear tests (regulation from x_T(0) = 0.5 m and
the constant trolley-side disturbance of -0.35 kN, discrete design model, 30 s, +-10 kN saturation)
and with the same quantities: J = ITAE_r + ITAE_d, peak unsaturated command, overshoot, peak swing and
minimum closed-loop damping ratio. The MPC is represented by its terminal law u = -K_zeta [x; u_prev]
(identical to the MPC in the nominal cases)."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sim import params as P, setup, controllers as C, tuning as TU
from sim.linear import z_to_s, damping_ratios


def run(law, x0, d, n=int(round(P.T_END / P.TS)), Fmax=P.F_MAX):
    _, _, Ad, Bd = setup.design_model()
    x, u_prev, X, Uun = np.array(x0, float), 0.0, [], []
    for _ in range(n + 1):
        X.append(x.copy())
        u_un = law(x, u_prev)
        u = min(max(u_un, -Fmax), Fmax)
        Uun.append(u_un)
        x = Ad @ x + Bd[:, 0] * (u + d)
        u_prev = u
    return np.array(X), np.array(Uun)


def evaluate(law, Acl):
    t = np.arange(int(round(P.T_END / P.TS)) + 1) * P.TS
    X, Uun = run(law, [0.5, 0, 0, 0], 0.0)
    Xd, _ = run(law, [0, 0, 0, 0], TU.D_TEST)
    ev = np.linalg.eigvals(Acl)
    return dict(ITAE=float(np.trapezoid(t * np.abs(X[:, 0]), t)), ITAE_d=float(np.trapezoid(t * np.abs(Xd[:, 0]), t)),
                Fun=float(np.max(np.abs(Uun))), OS=float(max(0.0, -X[:, 0].min()) / 0.5 * 100),
                PHI=float(np.degrees(np.max(np.abs(X[:, 2] / P.L_ROPE)))),
                zeta=float(np.min(damping_ratios(z_to_s(ev, P.TS)))), rho=float(np.max(np.abs(ev))))


if __name__ == '__main__':
    _, _, Ad, Bd = setup.design_model()
    K, _, _ = setup.lqr_gain()
    K = K.ravel()
    out = {'LQR': evaluate(lambda x, up: float(-K @ x), Ad - Bd @ K[None, :])}
    _, Ka, Aa, Ba = C.mpc_terminal_ingredients(Ad, Bd, setup.Q_MPC, setup.R_MPC, setup.RDU_MPC)
    Ka = Ka.ravel()
    out['MPC'] = evaluate(lambda x, up: float(-Ka[:4] @ x - Ka[4] * up), Aa - Ba @ Ka[None, :])
    for k, r in out.items():
        r['J'] = r['ITAE'] + r['ITAE_d']
        print(k, {q: round(v, 4) for q, v in r.items()})
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'lq_specs.json'), 'w'), indent=1)
