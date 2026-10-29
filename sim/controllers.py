"""Discrete-time controllers. Measurements z = [x_T, dx_T, x_CoM, dx_CoM] (SI), output u in kN.

All controllers share the same sampling period, the same +-F_MAX saturation and receive the
same measurements. The reference is x_T,ref = 0 (regulation), as in the manuscript.
"""
import time
import numpy as np
from scipy.optimize import lsq_linear

from . import params as P


class PID:
    """Two-channel parallel PID (PD if Ki = 0) with conditional-integration anti-windup.

    u = KpT e_T + KiT xi_T + KdT de_T + Kpa alpha + Kia xi_a + Kda dalpha,
    e_T = x_ref - x_T, alpha = x_CoM / l_nom. Derivative terms use the measured velocities.
    Anti-windup: (i) an integrator is frozen while the actuator is saturated and integrating
    would drive the command further into saturation; (ii) each integral contribution is
    bounded by the actuator rating, |Ki xi| <= F_max, i.e. |xi_T| <= F_max/|KiT| (m s) and
    |xi_a| <= F_max/|Kia| (rad s).
    """

    def __init__(self, gains, l_nom=P.L_ROPE, Ts=P.TS, Fmax=P.F_MAX, name='PID'):
        self.KpT, self.KiT, self.KdT, self.Kpa, self.Kia, self.Kda = [float(g) for g in gains]
        self.l, self.Ts, self.Fmax, self.name = l_nom, Ts, Fmax, name
        self.reset()

    def reset(self):
        self.xiT = 0.0
        self.xia = 0.0

    def __call__(self, z, x_ref=0.0):
        eT, deT = x_ref - z[0], -z[1]
        al, dal = z[2] / self.l, z[3] / self.l
        u_un = (self.KpT * eT + self.KiT * self.xiT + self.KdT * deT
                + self.Kpa * al + self.Kia * self.xia + self.Kda * dal)
        u = float(np.clip(u_un, -self.Fmax, self.Fmax))
        sat = abs(u_un) > self.Fmax + P.SAT_TOL
        for ch, (Ki, e) in enumerate(((self.KiT, eT), (self.Kia, al))):
            if Ki == 0.0:
                continue
            if sat and np.sign(Ki * e) == np.sign(u_un):
                continue                       # conditional integration (clamping)
            bound = self.Fmax / abs(Ki)
            if ch == 0:
                self.xiT = float(np.clip(self.xiT + self.Ts * e, -bound, bound))
            else:
                self.xia = float(np.clip(self.xia + self.Ts * e, -bound, bound))
        return u, u_un


class StateFeedback:
    """u = -K z, saturated (used for the discrete-time LQR)."""

    def __init__(self, K, Fmax=P.F_MAX, name='LQR'):
        self.K = np.asarray(K, float).ravel()
        self.Fmax, self.name = Fmax, name

    def reset(self):
        pass

    def __call__(self, z, x_ref=0.0):
        zz = np.array(z, float)
        zz[0] -= x_ref
        u_un = float(-self.K @ zz)
        return float(np.clip(u_un, -self.Fmax, self.Fmax)), u_un


class AugmentedLQR:
    """Saturated LQR on the augmented state [x; u_prev] (stage cost with a rate penalty).
    Used to test whether the input-constrained MPC differs from saturated state feedback."""

    def __init__(self, K_aug, Fmax=P.F_MAX, name='Saturated LQR (MPC weights)'):
        self.K = np.asarray(K_aug, float).ravel()
        self.Fmax, self.name = Fmax, name
        self.reset()

    def reset(self):
        self.u_prev = 0.0

    def __call__(self, z, x_ref=0.0):
        xi = np.concatenate([np.array(z, float) - np.array([x_ref, 0, 0, 0]), [self.u_prev]])
        u_un = float(-self.K @ xi)
        u = float(np.clip(u_un, -self.Fmax, self.Fmax))
        self.u_prev = u
        return u, u_un


def mpc_terminal_ingredients(Ad, Bd, Q, R, Rdu):
    """Terminal cost P_aug and terminal LQR gain K_aug on xi = [x; u_prev] from the DARE of the
    MPC stage cost x'Qx + R u^2 + Rdu (u - u_prev)^2 (cross term handled exactly)."""
    from .linear import dlqr
    n = Ad.shape[0]
    Aa = np.zeros((n + 1, n + 1))
    Aa[:n, :n] = Ad
    Ba = np.vstack([Bd, [[1.0]]])
    if Rdu > 0:
        Qa = np.zeros((n + 1, n + 1))
        Qa[:n, :n] = Q
        Qa[n, n] = Rdu
        Sa = np.zeros((n + 1, 1))
        Sa[n, 0] = -Rdu
        Ka, Pa, ev = dlqr(Aa, Ba, Qa, np.array([[R + Rdu]]), S=Sa)
    else:
        K, Pm, ev = dlqr(Ad, Bd, Q, np.array([[R]]))
        Ka = np.hstack([K, [[0.0]]])
        Pa = np.zeros((n + 1, n + 1))
        Pa[:n, :n] = Pm
    return Pa, Ka, Aa, Ba


class MPC:
    """Linear MPC with terminal cost (condensed QP, all N moves free).

    min_U  sum_{k=0}^{N-1} [x_k' Q x_k + R u_k^2 + Rdu (u_k - u_{k-1})^2] + xi_N' P_aug xi_N
    s.t.   x_{k+1} = Ad x_k + Bd u_k,  |u_k| <= F_max,
           optionally |x_CoM,k| <= l*alpha_max and |dx_T,k| <= v_max (k = 1..N, softened by two
           non-negative slacks shared over the horizon, i.e. infinity-norm softening, with an exact
           linear + quadratic penalty),
    xi_N = [x_N; u_{N-1}]; P_aug from the DARE of the same stage cost, i.e. the unconstrained
    infinite-horizon cost-to-go of the terminal LQR law u = -K_aug xi.
    Input-constrained problem: exact BVLS (SciPy lsq_linear). With state constraints: primal-dual
    interior-point method (sim/qp.py).
    """

    def __init__(self, Ad, Bd, Q, R, Rdu, N, Fmax=P.F_MAX, name='MPC', l_nom=P.L_ROPE,
                 alpha_max=None, v_max=None, term_set=None, rho1=1e7, rho2=1e5):
        from .qp import ipm_qp
        self._ipm = ipm_qp
        n = Ad.shape[0]
        self.Ad, self.Bd, self.N, self.Fmax, self.name, self.Rdu = Ad, Bd, N, Fmax, name, Rdu
        Pa, Ka, Aa, Ba = mpc_terminal_ingredients(Ad, Bd, Q, R, Rdu)
        self.P_aug, self.K_aug, self.A_aug, self.B_aug = Pa, Ka, Aa, Ba
        self.term_set = term_set
        Phi = np.zeros((N * n, n))
        Gam = np.zeros((N * n, N))
        powers = [np.eye(n)]
        for i in range(N):
            powers.append(powers[-1] @ Ad)
            Phi[i * n:(i + 1) * n] = powers[-1]
        for i in range(N):
            for j in range(i + 1):
                Gam[i * n:(i + 1) * n, j] = (powers[i - j] @ Bd)[:, 0]
        self.Phi, self.Gam = Phi, Gam
        # quadratic cost J = U'HU + 2 f'U + const,  f = Fx x0 + Fu u_prev
        Qb = np.zeros((N * n, N * n))
        for i in range(N - 1):
            Qb[i * n:(i + 1) * n, i * n:(i + 1) * n] = Q
        Qb[(N - 1) * n:, (N - 1) * n:] = Pa[:n, :n]
        H = Gam.T @ Qb @ Gam + float(R) * np.eye(N)
        Fx = Gam.T @ Qb @ Phi
        Fu = np.zeros(N)
        D = np.eye(N) - np.eye(N, k=-1)
        H += Rdu * D.T @ D
        Fu[0] = -Rdu
        eN = np.zeros(N)
        eN[-1] = 1.0
        GN, PhN = Gam[(N - 1) * n:, :], Phi[(N - 1) * n:, :]
        pxu, puu = Pa[:n, n], Pa[n, n]
        H += np.outer(GN.T @ pxu, eN) + np.outer(eN, GN.T @ pxu) + puu * np.outer(eN, eN)
        Fx = Fx + np.outer(eN, pxu @ PhN)
        self.H = 0.5 * (H + H.T)
        self.Fx, self.Fu = Fx, Fu
        self.L = np.linalg.cholesky(self.H)
        # optional softened state constraints
        self.state_con = alpha_max is not None or v_max is not None
        if self.state_con:
            rows, lims, which = [], [], []
            for k in range(N):
                if alpha_max is not None:
                    rows.append(k * n + 2); lims.append(l_nom * np.sin(alpha_max)); which.append(0)
                if v_max is not None:
                    rows.append(k * n + 1); lims.append(v_max); which.append(1)
            Sx = np.zeros((len(rows), N * n))
            Sx[np.arange(len(rows)), rows] = 1.0
            self.SxG, self.SxP = Sx @ Gam, Sx @ Phi
            self.lims, self.which = np.array(lims), np.array(which)
            self.rho1, self.rho2 = rho1, rho2
        self.reset()

    def reset(self):
        self.u_prev = 0.0
        self.solve_times, self.term_ok, self.n_active, self.slack, self.iters = [], [], [], [], []
        self.fallbacks = 0
        self._last = None

    def _solve_box(self, f):
        d = -np.linalg.solve(self.L, f)
        # the SciPy default iteration cap of BVLS (number of variables) is too small for the saturated
        # start of S2; a non-converged solve is an error, not a silently returned plan
        res = lsq_linear(self.L.T, d, bounds=(-self.Fmax, self.Fmax), method='bvls', tol=1e-12,
                         max_iter=20 * self.N)
        if res.status <= 0:
            raise RuntimeError(f'BVLS did not converge (status {res.status}, {res.nit} iterations)')
        return res.x, res.nit

    def _solve_state(self, f, x0):
        N = self.N
        nv = N + 2
        Hq = np.zeros((nv, nv))
        Hq[:N, :N] = 2.0 * self.H
        Hq[N, N] = Hq[N + 1, N + 1] = 2.0 * self.rho2
        fq = np.concatenate([2.0 * f, [self.rho1, self.rho1]])
        sx = self.SxP @ x0
        E = np.zeros((len(self.lims), 2))
        E[np.arange(len(self.lims)), self.which] = -1.0
        G = np.vstack([np.hstack([np.eye(N), np.zeros((N, 2))]),
                       np.hstack([-np.eye(N), np.zeros((N, 2))]),
                       np.hstack([self.SxG, E]),
                       np.hstack([-self.SxG, E]),
                       np.hstack([np.zeros((2, N)), -np.eye(2)])])
        h = np.concatenate([np.full(N, self.Fmax), np.full(N, self.Fmax),
                            self.lims - sx, self.lims + sx, np.zeros(2)])
        sol, it, ok = self._ipm(Hq, fq, G, h)
        if not ok:                                   # restart from the shifted previous plan
            x0q = np.concatenate([np.clip(np.append(self._last[1:], self._last[-1]), -self.Fmax, self.Fmax),
                                  [1.0, 1.0]]) if self._last is not None else None
            sol, it2, ok = self._ipm(Hq, fq, G, h, x0=x0q)
            it += it2
        if not ok or not np.all(np.isfinite(sol)):   # last resort: input-constrained solution
            self.fallbacks += 1
            U, _ = self._solve_box(f)
            self.slack.append(np.array([np.nan, np.nan]))
            self._last = U
            return U, it
        self.slack.append(sol[N:].copy())
        self._last = sol[:N].copy()
        return sol[:N], it

    def __call__(self, z, x_ref=0.0):
        x0 = np.array(z, float)
        x0[0] -= x_ref
        f = self.Fx @ x0 + self.Fu * self.u_prev
        t0 = time.perf_counter()
        U, it = self._solve_state(f, x0) if self.state_con else self._solve_box(f)
        self.solve_times.append(time.perf_counter() - t0)
        self.iters.append(it)
        self.n_active.append(int(np.sum(np.abs(U) > self.Fmax - 1e-6)))
        if self.term_set is not None:
            xN = self.Phi[-4:] @ x0 + self.Gam[-4:] @ U
            self.term_ok.append(self.term_set.contains(np.concatenate([xN, [U[-1]]])))
        u = float(np.clip(U[0], -self.Fmax, self.Fmax))
        self.u_prev = u
        return u, u
