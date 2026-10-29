"""Closed-loop simulation on the nonlinear evaluation plant or on the linear design model."""
import numpy as np

from . import params as P
from . import plant
from .linear import c2d


class Scenario:
    """Initial condition, plant parameters and disturbance of one simulation run."""

    def __init__(self, name, xT0=0.5, phi0_deg=0.0, y0=0.0, theta0_deg=0.0, prm=None,
                 wind_x=0.0, wind_t=(0.0, 1e9), T=P.T_END, ref_rate=None):
        self.name, self.xT0, self.phi0, self.y0 = name, xT0, np.radians(phi0_deg), y0
        self.ref_rate = ref_rate          # optional rate-limited reference ramp x_ref(t), m/s
        self.theta0 = np.radians(theta0_deg)
        self.prm = P.plant_params() if prm is None else prm
        self.wind_x, self.wind_t, self.T = wind_x, wind_t, T

    def x_ref(self, t):
        if self.ref_rate is None:
            return 0.0
        return np.sign(self.xT0) * max(abs(self.xT0) - self.ref_rate * t, 0.0)

    def x_com0(self):
        return self.prm[2] * np.sin(self.phi0)


def run_nonlinear(ctrl, sc, Ts=P.TS, h=P.H_INT):
    prm = sc.prm
    s = plant.initial_state(prm, xT=sc.xT0, x_com=sc.x_com0(), y_com=sc.y0, theta=sc.theta0)
    ctrl.reset()
    n = int(round(sc.T / Ts))
    log = {k: np.zeros(n + 1) for k in
           ('t', 'xT', 'vT', 'xc', 'vc', 'yc', 'theta', 'phi', 'u', 'u_un', 'Fact', 'Trope')}
    work = 0.0
    log['Trope'][0] = 0.5 * prm[1] * prm[4]
    for k in range(n + 1):
        t = k * Ts
        z = plant.measured(s)
        log['t'][k], log['xT'][k], log['vT'][k] = t, s[0], s[6]
        log['xc'][k], log['vc'][k] = z[2], z[3]
        log['yc'][k], log['theta'][k], log['phi'][k] = s[2], s[4], plant.swing_angle(s)
        log['Fact'][k] = s[12] if prm[11] > 0 else np.nan
        if k == n:
            log['u'][k], log['u_un'][k] = log['u'][k - 1], log['u_un'][k - 1]
            break
        u, u_un = ctrl(z, sc.x_ref(t))
        log['u'][k], log['u_un'][k] = u, u_un
        Fwx = sc.wind_x if sc.wind_t[0] <= t < sc.wind_t[1] else 0.0
        s, tmax, dw = plant.step_zoh(s, 1000.0 * u, Fwx, 0.0, prm, Ts, h)
        work += dw
        log['Trope'][k + 1] = tmax
        if not np.all(np.isfinite(s)) or abs(s[0] - sc.xT0) > 1e3:
            raise FloatingPointError('plant diverged')
    log['work'] = work
    r1, r2 = plant.constraint_residual(s, prm)
    log['constraint_residual'] = max(abs(r1), abs(r2))
    return log


def run_linear(ctrl, sc, Ts=P.TS, m1=P.M1, m2=P.M2, l=P.L_ROPE):
    """Same protocol on the linear design model (exact ZOH, frictionless, no disturbance)."""
    from .params import linear_model
    A, B, _, _ = linear_model(m1, m2, l)
    nsub = 40
    Ad_f, Bd_f = c2d(A, B, Ts / nsub)
    x = np.array([sc.xT0, 0.0, l * np.sin(sc.phi0), 0.0])
    ctrl.reset()
    n = int(round(sc.T / Ts))
    log = {k: np.zeros(n + 1) for k in ('t', 'xT', 'vT', 'xc', 'vc', 'phi', 'u', 'u_un')}
    work = 0.0
    for k in range(n + 1):
        log['t'][k], log['xT'][k], log['vT'][k] = k * Ts, x[0], x[1]
        log['xc'][k], log['vc'][k], log['phi'][k] = x[2], x[3], np.arcsin(np.clip(x[2] / l, -1, 1))
        if k == n:
            log['u'][k], log['u_un'][k] = log['u'][k - 1], log['u_un'][k - 1]
            break
        u, u_un = ctrl(x.copy())
        log['u'][k], log['u_un'][k] = u, u_un
        for _ in range(nsub):
            v0 = x[1]
            x = Ad_f @ x + Bd_f[:, 0] * u
            work += 0.5 * (Ts / nsub) * 1000.0 * abs(u) * (abs(v0) + abs(x[1]))
    log['work'] = work
    log['Trope'] = np.full(n + 1, np.nan)
    log['yc'] = np.zeros(n + 1)
    log['theta'] = np.zeros(n + 1)
    return log
