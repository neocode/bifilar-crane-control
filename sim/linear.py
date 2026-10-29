"""Linear design-model utilities: ZOH discretisation, discrete LQR, PID closed loops."""
import numpy as np
from scipy.linalg import expm, solve_discrete_are


def c2d(A, B, Ts):
    """Exact zero-order-hold discretisation."""
    n, m = A.shape[0], B.shape[1]
    Mx = np.zeros((n + m, n + m))
    Mx[:n, :n] = A
    Mx[:n, n:] = B
    E = expm(Mx * Ts)
    return E[:n, :n], E[:n, n:]


def dlqr(Ad, Bd, Q, R, S=None):
    """Discrete LQR: returns K (u = -K x), P (DARE solution) and closed-loop eigenvalues."""
    R = np.atleast_2d(R)
    if S is None:
        S = np.zeros((Ad.shape[0], Bd.shape[1]))
    P = solve_discrete_are(Ad, Bd, Q, R, s=S)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad + S.T)
    return K, P, np.linalg.eigvals(Ad - Bd @ K)


def z_to_s(z, Ts):
    return np.log(z.astype(complex)) / Ts


def pid_closed_loop(Ad, Bd, gains, l, Ts):
    """Linear (unsaturated) closed loop of the discrete PID (forward-Euler integrators).

    gains = (KpT, KiT, KdT, Kpa, Kia, Kda), control law (kN)
        u = KpT e_T + KiT xi_T + KdT de_T + Kpa alpha + Kia xi_a + Kda dalpha,
    e_T = -x_T, alpha = x_CoM / l, xi_T(k+1) = xi_T(k) + Ts e_T, xi_a(k+1) = xi_a(k) + Ts alpha.
    Augmented state [x(4), xi_T, xi_a]. Returns the 6x6 closed-loop matrix and u = Kaug z.
    """
    KpT, KiT, KdT, Kpa, Kia, Kda = gains
    Kx = np.array([-KpT, -KdT, Kpa / l, Kda / l])
    Kaug = np.concatenate([Kx, [KiT, Kia]])
    Acl = np.zeros((6, 6))
    Acl[:4, :4] = Ad + Bd @ Kx[None, :]
    Acl[:4, 4] = Bd[:, 0] * KiT
    Acl[:4, 5] = Bd[:, 0] * Kia
    Acl[4, :] = Ts * np.array([-1.0, 0, 0, 0, 0, 0]) + np.array([0, 0, 0, 0, 1.0, 0])
    Acl[5, :] = Ts * np.array([0, 0, 1.0 / l, 0, 0, 0]) + np.array([0, 0, 0, 0, 0, 1.0])
    return Acl, Kaug


def damping_ratios(eig_s):
    """Damping ratios of continuous-equivalent poles (real poles give 1, unstable < 0)."""
    zs = []
    for p in eig_s:
        w = abs(p)
        zs.append(-p.real / w if w > 1e-9 else 1.0)
    return np.array(zs)
