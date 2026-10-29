"""Specification-based tuning of the PD/PID controllers on the linear design model.

Each controller variant minimises J = ITAE_sp + ITAE_d over the 30 s horizon, where ITAE_sp is
the trolley-position ITAE of the regulation test (x_T(0) = 0.5 m, all other states zero) and
ITAE_d that of a load-disturbance test (system initially at rest, constant force D_TEST acting
on the trolley, representing the travel resistance). The minimisation is subject to a set of
technical specifications evaluated on the regulation test (peak unsaturated command, overshoot,
peak swing) and on the linear closed loop (minimum damping ratio). The closed loop is simulated
in discrete time (exact ZOH model, sampling period TS) with the same saturation and anti-windup
logic as the evaluation runs (sim/controllers.py:PID).
"""
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, minimize

from . import params as P
from .linear import c2d, pid_closed_loop, z_to_s, damping_ratios, pid_structural_mode, remove_structural_mode


D_TEST = -0.35   # kN, constant trolley-side disturbance of the tuning test (~ nominal Coulomb force)


@njit(cache=True)
def pid_sim_linear(g, Ad, Bd, l, Ts, x0, n, Fmax, d=0.0):
    KpT, KiT, KdT, Kpa, Kia, Kda = g[0], g[1], g[2], g[3], g[4], g[5]
    bcol = Bd[:, 0].copy()
    x = x0.copy()
    xiT = 0.0
    xia = 0.0
    X = np.zeros((n + 1, 4))
    U = np.zeros(n + 1)
    Uun = np.zeros(n + 1)
    for k in range(n + 1):
        X[k] = x
        eT, deT = -x[0], -x[1]
        al, dal = x[2] / l, x[3] / l
        u_un = KpT * eT + KiT * xiT + KdT * deT + Kpa * al + Kia * xia + Kda * dal
        u = min(max(u_un, -Fmax), Fmax)
        sat = abs(u_un) > Fmax
        U[k] = u
        Uun[k] = u_un
        if KiT != 0.0:
            if not (sat and np.sign(KiT * eT) == np.sign(u_un)):
                b = Fmax / abs(KiT)
                xiT = min(max(xiT + Ts * eT, -b), b)
        if Kia != 0.0:
            if not (sat and np.sign(Kia * al) == np.sign(u_un)):
                b = Fmax / abs(Kia)
                xia = min(max(xia + Ts * al, -b), b)
        x = Ad @ x + bcol * (u + d)
        if not np.isfinite(x[0]) or abs(x[0]) > 1e3:
            return X[:k + 1], U[:k + 1], Uun[:k + 1], False
    return X, U, Uun, True


# technical specifications of the four PID variants (and of the PD controller)
# F_spec : maximum unsaturated command for x_T(0)=0.5 m (kN); None = saturation permitted
# OS     : maximum trolley overshoot (% of the initial offset); None = not constrained
# PHI    : maximum swing angle (deg); None = not constrained
# ZETA   : minimum damping ratio of the linear closed-loop poles
SPECS = {
    'PID (soft)':    dict(F_spec=2.5, OS=5.0, PHI=0.5, ZETA=0.70),
    'PID (medium)':  dict(F_spec=5.0, OS=10.0, PHI=1.0, ZETA=0.50),
    'PID (optimal)': dict(F_spec=10.0, OS=10.0, PHI=1.5, ZETA=0.40),
    'PID (hard)':    dict(F_spec=None, OS=20.0, PHI=3.0, ZETA=0.20),
    'PD':            dict(F_spec=10.0, OS=10.0, PHI=1.5, ZETA=0.40),
}

BOUNDS = [(0.0, 200.0),      # KpT  kN/m
          (0.0, 200.0),      # KiT  kN/(m s)
          (0.0, 400.0),      # KdT  kN s/m
          (-3000.0, 3000.0),  # Kpa kN/rad
          (-1000.0, 1000.0),  # Kia kN/(rad s)
          (-3000.0, 3000.0)]  # Kda kN s/rad


class Tuner:
    def __init__(self, spec, integral=True, swing_integral=True, x0=0.5, Ts=P.TS, T=P.T_END,
                 Fmax=P.F_MAX):
        """integral=False: PD; swing_integral=False: integral action in the trolley channel only."""
        A, B, _, _ = P.linear_model()
        Ad, Bd = c2d(A, B, Ts)
        self.Ad, self.Bd = np.ascontiguousarray(Ad), np.ascontiguousarray(Bd)
        self.l, self.Ts, self.Fmax, self.x0 = P.L_ROPE, Ts, Fmax, x0
        self.w_struct = pid_structural_mode(self.Ad, self.Bd, self.l, Ts)
        self.n = int(round(T / Ts))
        self.t = np.arange(self.n + 1) * Ts
        self.spec, self.integral, self.swing_integral = spec, integral, swing_integral

    def free(self):
        """Indices of the free gains in (KpT, KiT, KdT, Kpa, Kia, Kda)."""
        if not self.integral:
            return [0, 2, 3, 5]
        return [0, 1, 2, 3, 5] if not self.swing_integral else [0, 1, 2, 3, 4, 5]

    def full(self, p):
        g = np.zeros(6)
        g[self.free()] = np.array(p, float)
        return g

    def evaluate(self, g):
        X, U, Uun, ok = pid_sim_linear(g, self.Ad, self.Bd, self.l, self.Ts,
                                       np.array([self.x0, 0, 0, 0.0]), self.n, self.Fmax, 0.0)
        Xd, Ud, Uund, okd = pid_sim_linear(g, self.Ad, self.Bd, self.l, self.Ts,
                                           np.zeros(4), self.n, self.Fmax, D_TEST)
        Acl, _ = pid_closed_loop(self.Ad, self.Bd, g, self.l, self.Ts)
        keep = [0, 1, 2, 3] + ([4] if g[1] != 0.0 else []) + ([5] if g[4] != 0.0 else [])
        Acl = Acl[np.ix_(keep, keep)]
        if g[4] != 0.0:
            # the swing-angle integrator adds an uncontrollable eigenvalue z = 1 for any gains
            # (Sect. 3.4.4); it is removed analytically, so that stability (rho < 1) and the
            # damping specification refer to the remaining modes and do not depend on rounding
            Acl = remove_structural_mode(Acl, self.w_struct[keep])
        ev = np.linalg.eigvals(Acl)
        rho = np.max(np.abs(ev))
        zeta = np.min(damping_ratios(z_to_s(ev, self.Ts)))
        r = dict(ok=ok and okd and rho < 1.0, rho=rho, zeta=zeta)
        if not r['ok']:
            return r
        t = self.t
        xT = X[:, 0]
        r['ITAE'] = np.trapezoid(t * np.abs(xT), t)
        r['ITAE_d'] = np.trapezoid(t * np.abs(Xd[:, 0]), t)
        r['IAE'] = np.trapezoid(np.abs(xT), t)
        r['Fun'] = np.max(np.abs(Uun))
        r['OS'] = max(0.0, -np.min(xT)) / self.x0 * 100
        r['PHI'] = np.degrees(np.max(np.abs(X[:, 2] / self.l)))
        r['ed_final_mm'] = abs(Xd[-1, 0]) * 1000
        out = np.where(np.abs(xT) > 0.05 * self.x0)[0]
        r['Ts'] = np.inf if (len(out) and out[-1] == len(xT) - 1) else (t[out[-1] + 1] if len(out) else 0)
        return r

    def cost(self, p):
        g = self.full(p)
        r = self.evaluate(g)
        if not r['ok']:
            return 1e6 + 1e4 * r['rho']
        s = self.spec
        v = max(0.0, s['ZETA'] - r['zeta']) / 0.05
        if s['F_spec'] is not None:
            v += max(0.0, r['Fun'] - s['F_spec']) / 0.5
        if s['OS'] is not None:
            v += max(0.0, r['OS'] - s['OS']) / 1.0
        if s['PHI'] is not None:
            v += max(0.0, r['PHI'] - s['PHI']) / 0.05
        if not np.isfinite(r['Ts']):
            v += 1.0
        return r['ITAE'] + r['ITAE_d'] + 100.0 * v

    def run(self, seed=1, maxiter=1000, popsize=50, x0=None):
        bounds = [BOUNDS[i] for i in self.free()]
        if x0 is not None and len(x0) == 6:
            x0 = np.asarray(x0, float)[self.free()]
        de = differential_evolution(self.cost, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
                                    tol=1e-12, mutation=(0.5, 1.0), recombination=0.7,
                                    polish=False, init='sobol', updating='deferred', workers=1,
                                    x0=x0)
        best_p, best_f = de.x, de.fun
        for _ in range(3):                      # restarted Nelder-Mead refinement
            loc = minimize(self.cost, best_p, method='Nelder-Mead',
                           options=dict(xatol=1e-8, fatol=1e-10, maxiter=40000, maxfev=80000,
                                        adaptive=True))
            if loc.fun < best_f - 1e-12:
                best_p, best_f = loc.x, loc.fun
            else:
                break
        g = self.full(best_p)
        return g, self.evaluate(g), best_f
