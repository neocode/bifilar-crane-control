"""Linear stability margins of the unsaturated closed loops (results/linear_margins.json).

For every controller, designed for the nominal parameters, the discrete closed loop of the linear
design model is formed (i) with the true rope length l in [4, 8] m while the controller keeps
l_nom = 6 m (swing feedback alpha = x_CoM / l_nom), and (ii) with a first-order drive lag tau in
[0, 0.2] s at the nominal rope length. The script reports the rope lengths and the smallest lag at
which a closed-loop eigenvalue leaves the unit circle. The structural eigenvalue z = 1 of the PID
loops (Sect. 3.4.4), which persists for any l and tau, is excluded. The input-constrained MPC is
represented by its terminal law u = -K_zeta [x; u_prev], with which it coincides when the input
constraint is inactive.
"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.linalg import expm
from sim import params as P, setup, controllers as C

KEYS = ['KpT', 'KiT', 'KdT', 'Kpa', 'Kia', 'Kda']
TS, LNOM = P.TS, P.L_ROPE


def plant(l=LNOM, tau=0.0):
    """ZOH model of the design model with rope length l and optional first-order lag (state F)."""
    A, B, _, _ = P.linear_model(l=l)
    if tau > 0:
        A = np.block([[A, B], [np.zeros((1, 4)), -np.ones((1, 1)) / tau]])
        B = np.vstack([np.zeros((4, 1)), np.ones((1, 1)) / tau])
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * TS)
    return E[:n, :n], E[:n, n:]


def closed_loop(name, l=LNOM, tau=0.0):
    Ad, Bd = plant(l, tau)
    n = Ad.shape[0]
    sel = np.zeros((4, n))
    sel[:, :4] = np.eye(4)                 # the controllers measure x only (not the lagged force)
    if name in ('PD', 'PID (soft)', 'PID (medium)', 'PID (optimal)', 'PID (hard)'):
        g = setup.load_gains()[name]['gains']
        KpT, KiT, KdT, Kpa, Kia, Kda = [g[k] for k in KEYS]
        Kx = np.array([-KpT, -KdT, Kpa / LNOM, Kda / LNOM]) @ sel
        Acl = np.zeros((n + 2, n + 2))
        Acl[:n, :n] = Ad + Bd @ Kx[None, :]
        Acl[:n, n], Acl[:n, n + 1] = Bd[:, 0] * KiT, Bd[:, 0] * Kia
        Acl[n, :n], Acl[n, n] = -TS * sel[0], 1.0
        Acl[n + 1, :n], Acl[n + 1, n + 1] = TS * sel[2] / LNOM, 1.0
        keep = list(range(n)) + ([n] if KiT else []) + ([n + 1] if Kia else [])
        return Acl[np.ix_(keep, keep)], bool(Kia)
    if name == 'LQR':
        K, _, _ = setup.lqr_gain()
        return Ad - Bd @ (K @ sel), False
    # MPC: terminal law on the augmented state [x; u_prev]
    _, _, Ad0, Bd0 = setup.design_model()
    _, Ka, _, _ = C.mpc_terminal_ingredients(Ad0, Bd0, setup.Q_MPC, setup.R_MPC, setup.RDU_MPC)
    Kx, Ku = (Ka[0, :4] @ sel), Ka[0, 4]
    Acl = np.zeros((n + 1, n + 1))
    Acl[:n, :n] = Ad - Bd @ Kx[None, :]
    Acl[:n, n] = -Bd[:, 0] * Ku
    Acl[n, :n], Acl[n, n] = -Kx, -Ku
    return Acl, False


def rho(name, l=LNOM, tau=0.0):
    Acl, structural = closed_loop(name, l, tau)
    ev = np.linalg.eigvals(Acl)
    if structural:
        i = np.argmin(np.abs(ev - 1.0))
        assert abs(ev[i] - 1.0) < 1e-8, 'structural eigenvalue not found'
        ev = np.delete(ev, i)
    k = np.argmax(np.abs(ev))
    return float(np.abs(ev[k])), ev[k]


if __name__ == '__main__':
    out = {}
    ls = np.round(np.arange(4.0, 8.0001, 0.01), 2)
    taus = np.round(np.arange(0.0, 0.2001, 0.001), 3)
    for name in setup.CLASSICAL + ['MPC']:
        r_l = np.array([rho(name, l=l)[0] for l in ls])
        unst = ls[r_l >= 1.0]
        r_t = np.array([rho(name, tau=t)[0] for t in taus])
        unst_t = taus[r_t >= 1.0]
        z8 = rho(name, l=8.0)[1]
        s8 = np.log(complex(z8)) / TS
        out[name] = dict(l_unstable_from=float(unst.min()) if len(unst) else None,
                         l_unstable_to=float(unst.max()) if len(unst) else None,
                         rho_l8=float(r_l[-1]), s_l8=[float(s8.real), float(abs(s8.imag))],
                         tau_crit=float(unst_t.min()) if len(unst_t) else None)
        print(name, out[name])
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'linear_margins.json'), 'w'),
              indent=1)
