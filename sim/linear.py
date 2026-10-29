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


def pid_structural_mode(Ad, Bd, l, Ts):
    """Left eigenvector w of the PID closed loop (w' Acl = w') for the eigenvalue 1 that the
    swing-angle integrator introduces for any gains. Its state ordering is that of
    pid_closed_loop, [x(4), xi_T, xi_a]; w = [w_x, 0, 1] with w_x' Bd = 0 and
    w_x' (Ad - I) + (Ts/l) e_3' = 0, the discrete counterpart of the invariant
    g xi_a + dx_T/dt + dx_CoM/dt (Sect. 3.4.4). It does not depend on the gains."""
    Mx = np.zeros((5, 5))
    Mx[:4, :4] = (Ad - np.eye(4)).T
    Mx[2, 4] = Ts / l
    Mx[4, :4] = Bd[:, 0]
    v = np.linalg.svd(Mx)[2][-1]
    return np.array([v[0], v[1], v[2], v[3], 0.0, v[4]]) / v[4]


def remove_structural_mode(Acl, w):
    """Restriction of Acl to its invariant subspace ker(w'): all eigenvalues except the
    structural eigenvalue 1 of the left eigenvector w."""
    from scipy.linalg import null_space
    res = np.max(np.abs(w @ Acl - w))
    if res > 1e-9:
        raise ValueError(f'w is not a left eigenvector for the eigenvalue 1 (residual {res:.1e})')
    Q = null_space(w[None, :])
    return Q.T @ Acl @ Q


def damping_ratios(eig_s):
    """Damping ratios of continuous-equivalent poles (real poles give 1, unstable < 0)."""
    zs = []
    for p in eig_s:
        w = abs(p)
        zs.append(-p.real / w if w > 1e-9 else 1.0)
    return np.array(zs)
