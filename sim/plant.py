"""Nonlinear evaluation plant: trolley + two inextensible massless ropes + rigid slender beam.

Redundant coordinates p = [x_T, X, Y, Z, theta, psi]:
  x_T            trolley displacement along the rail (fixed-frame X axis),
  (X, Y, Z)      absolute position of the beam centre of mass C (Z measured upwards from the
                 trolley rope attachment level, so Z = -l at rest),
  theta, psi     yaw (rotation about the vertical) and pitch of the beam axis,
  e = [cos(psi)cos(theta), cos(psi)sin(theta), sin(psi)] is the unit vector along A->B.

Rope attachment points: E = (x_T - a, 0, 0), D = (x_T + a, 0, 0) on the traverse;
A = r_C - a e, B = r_C + a e on the beam. Holonomic constraints |A-E| = |B-D| = l.

Equations of motion (index-1 form with Baumgarte stabilisation):
  M(p) p'' + G(p)^T lam = Q(p, p', F),     G p'' = -c(p, p') - 2 b ġ - b^2 g,
where lam_i * l are the rope tensions. The relative CoM displacements used in the paper are
x_CoM = X - x_T, y_CoM = Y; the small-motion limit of this model is Eqs. (11)-(14).

Optional state 13: actual actuator force with first-order lag tau (if tau > 0).
"""
import numpy as np
from numba import njit

NSTATE = 13


@njit(cache=True)
def _kinematics(s, prm):
    a = prm[3]
    xT, X, Y, Z, th, ps = s[0], s[1], s[2], s[3], s[4], s[5]
    cth, sth, cps, sps = np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
    e = np.array([cps * cth, cps * sth, sps])
    e_th = np.array([-cps * sth, cps * cth, 0.0])
    e_ps = np.array([-sps * cth, -sps * sth, cps])
    dA = np.array([X - a * e[0] - xT + a, Y - a * e[1], Z - a * e[2]])
    dB = np.array([X + a * e[0] - xT - a, Y + a * e[1], Z + a * e[2]])
    JA = np.zeros((3, 6))
    JB = np.zeros((3, 6))
    JA[0, 0] = -1.0
    JB[0, 0] = -1.0
    for i in range(3):
        JA[i, 1 + i] = 1.0
        JB[i, 1 + i] = 1.0
        JA[i, 4] = -a * e_th[i]
        JB[i, 4] = a * e_th[i]
        JA[i, 5] = -a * e_ps[i]
        JB[i, 5] = a * e_ps[i]
    return cth, sth, cps, sps, dA, dB, JA, JB


@njit(cache=True)
def friction(v, prm):
    """Trolley travel resistance (N): viscous + tanh-smoothed Coulomb term."""
    return prm[7] * v + prm[6] * np.tanh(v / prm[8])


@njit(cache=True)
def rhs(s, Fcmd, Fwx, Fwy, prm):
    """Time derivative of the plant state. Fcmd, Fwx, Fwy in N."""
    m1, m2, l, a, g, J = prm[0], prm[1], prm[2], prm[3], prm[4], prm[5]
    cs, cth_d, tau, ab, bb = prm[9], prm[10], prm[11], prm[12], prm[13]
    cth, sth, cps, sps, dA, dB, JA, JB = _kinematics(s, prm)
    v = s[6:12]
    th_d, ps_d = v[4], v[5]

    Fact = s[12] if tau > 0.0 else Fcmd

    # mass matrix and generalised forces
    Mv = np.array([m1, m2, m2, m2, J * cps * cps, J])
    Q = np.zeros(6)
    rel = v[1] - v[0]
    Q[0] = Fact - friction(v[0], prm) + cs * rel
    Q[1] = -cs * rel + Fwx
    Q[2] = -cs * v[2] + Fwy
    Q[3] = -m2 * g - cs * v[3]
    Q[4] = 2.0 * J * cps * sps * ps_d * th_d - cth_d * th_d
    Q[5] = -J * cps * sps * th_d * th_d - cth_d * ps_d

    # constraint Jacobian and acceleration bias
    G = np.zeros((2, 6))
    for j in range(6):
        G[0, j] = dA[0] * JA[0, j] + dA[1] * JA[1, j] + dA[2] * JA[2, j]
        G[1, j] = dB[0] * JB[0, j] + dB[1] * JB[1, j] + dB[2] * JB[2, j]
    vA = JA @ v
    vB = JB @ v
    e_thth = np.array([-cps * cth, -cps * sth, 0.0])
    e_thps = np.array([sps * sth, -sps * cth, 0.0])
    e_psps = np.array([-cps * cth, -cps * sth, -sps])
    h = e_thth * th_d * th_d + 2.0 * e_thps * th_d * ps_d + e_psps * ps_d * ps_d
    c1 = vA @ vA - a * (dA @ h)
    c2 = vB @ vB + a * (dB @ h)
    g1 = 0.5 * (dA @ dA - l * l)
    g2 = 0.5 * (dB @ dB - l * l)
    gd1 = dA @ vA
    gd2 = dB @ vB

    K = np.zeros((8, 8))
    r = np.zeros(8)
    for i in range(6):
        K[i, i] = Mv[i]
        r[i] = Q[i]
    for j in range(6):
        K[j, 6] = G[0, j]
        K[j, 7] = G[1, j]
        K[6, j] = G[0, j]
        K[7, j] = G[1, j]
    r[6] = -c1 - 2.0 * ab * gd1 - bb * bb * g1
    r[7] = -c2 - 2.0 * ab * gd2 - bb * bb * g2
    sol = np.linalg.solve(K, r)

    ds = np.zeros(NSTATE)
    ds[0:6] = v
    ds[6:12] = sol[0:6]
    if tau > 0.0:
        ds[12] = (Fcmd - s[12]) / tau
    return ds, sol[6] * np.sqrt(dA @ dA), sol[7] * np.sqrt(dB @ dB)


@njit(cache=True)
def step_zoh(s, Fcmd, Fwx, Fwy, prm, T, h):
    """Advance the plant over one sampling interval T with RK4 sub-steps h (ZOH input).
    Returns the new state, the maximum rope tension (N) at the sub-step nodes and the
    absolute mechanical work increment int |F_act * dx_T/dt| dt (J, trapezoidal rule)."""
    n = int(np.round(T / h))
    tmax = 0.0
    work = 0.0
    tau = prm[11]
    p_old = abs((s[12] if tau > 0.0 else Fcmd) * s[6])
    for _ in range(n):
        k1, t1, t2 = rhs(s, Fcmd, Fwx, Fwy, prm)
        tmax = max(tmax, t1, t2)
        k2, _a, _b = rhs(s + 0.5 * h * k1, Fcmd, Fwx, Fwy, prm)
        k3, _a, _b = rhs(s + 0.5 * h * k2, Fcmd, Fwx, Fwy, prm)
        k4, _a, _b = rhs(s + h * k3, Fcmd, Fwx, Fwy, prm)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        p_new = abs((s[12] if tau > 0.0 else Fcmd) * s[6])
        work += 0.5 * h * (p_old + p_new)
        p_old = p_new
    return s, tmax, work


@njit(cache=True)
def constraint_residual(s, prm):
    l = prm[2]
    _c, _s, _c2, _s2, dA, dB, _JA, _JB = _kinematics(s, prm)
    return np.sqrt(dA @ dA) - l, np.sqrt(dB @ dB) - l


def initial_state(prm, xT=0.0, x_com=0.0, y_com=0.0, theta=0.0, F0=0.0):
    """Consistent initial state at rest for prescribed (x_T, x_CoM, y_CoM, theta).
    Solves the two rope constraints for the dependent coordinates (Z, psi)."""
    l, a = prm[2], prm[3]
    X, Y = xT + x_com, y_com
    Z, ps = -np.sqrt(max(l * l - x_com ** 2 - y_com ** 2, 1e-9)), 0.0
    for _ in range(50):
        s = np.zeros(NSTATE)
        s[:6] = [xT, X, Y, Z, theta, ps]
        r1, r2 = constraint_residual(s, prm)
        if abs(r1) + abs(r2) < 1e-13:
            break
        Jm = np.zeros((2, 2))
        eps = 1e-7
        for j, idx in enumerate((3, 5)):
            sp = s.copy()
            sp[idx] += eps
            q1, q2 = constraint_residual(sp, prm)
            Jm[0, j] = (q1 - r1) / eps
            Jm[1, j] = (q2 - r2) / eps
        dz = np.linalg.solve(Jm, -np.array([r1, r2]))
        Z += dz[0]
        ps += dz[1]
    s = np.zeros(NSTATE)
    s[:6] = [xT, X, Y, Z, theta, ps]
    s[12] = F0
    return s


def measured(s):
    """Signals available to the controllers: [x_T, dx_T, x_CoM, dx_CoM] (SI)."""
    return np.array([s[0], s[6], s[1] - s[0], s[7] - s[6]])


def swing_angle(s):
    """Rope angle in the longitudinal (x-z) plane, rad."""
    return np.arctan2(s[1] - s[0], -s[3])


def total_energy(s, prm):
    m1, m2, g, J = prm[0], prm[1], prm[4], prm[5]
    v = s[6:12]
    cps = np.cos(s[5])
    T = 0.5 * m1 * v[0] ** 2 + 0.5 * m2 * (v[1] ** 2 + v[2] ** 2 + v[3] ** 2) \
        + 0.5 * J * (v[5] ** 2 + (cps * v[4]) ** 2)
    return T + m2 * g * s[3]
