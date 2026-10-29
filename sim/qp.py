"""Dense convex QP solver: min 1/2 x'Hx + f'x  s.t.  G x <= h.

Primal-dual interior-point method with Mehrotra predictor-corrector steps (Nocedal & Wright,
Numerical Optimization, 2nd ed., Sect. 16.6). Used for the MPC variant with (softened) state
constraints; the input-constrained MPC uses the exact BVLS active-set solver from SciPy.
"""
import numpy as np


def ipm_qp(H, f, G, h, x0=None, tol=1e-10, max_iter=200):
    c = max(1.0, np.max(np.abs(np.diag(H))))     # objective scaling (solution unchanged)
    H, f = H / c, f / c
    n, m = H.shape[0], G.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, float)
    s = np.maximum(h - G @ x, 1.0)
    z = np.ones(m)
    scale = 1.0 + max(np.max(np.abs(f)), np.max(np.abs(h)))
    met = None
    best = (np.inf, x.copy())
    for it in range(max_iter):
        rd = H @ x + f + G.T @ z
        rp = G @ x + s - h
        mu = s @ z / m
        if not (np.all(np.isfinite(x)) and np.isfinite(mu)):
            break                                # numerical breakdown: return best iterate
        merit = np.linalg.norm(rd, np.inf) + np.linalg.norm(rp, np.inf) + mu
        if merit < best[0]:
            best = (merit, x.copy())
        if np.linalg.norm(rd, np.inf) < tol * scale and np.linalg.norm(rp, np.inf) < tol * scale \
                and mu < tol * scale:
            xp, ok = _polish(H, f, G, h, x, s, z)
            if ok:
                return xp, it, True
            met = (x.copy(), it) if met is None else met
            if it - met[1] >= 15:               # tolerance met but no clean active set: accept
                return x, it, True
        W = z / s
        Kmat = H + G.T @ (W[:, None] * G)
        try:
            Lc = np.linalg.cholesky(Kmat)
            lin = lambda b: np.linalg.solve(Lc.T, np.linalg.solve(Lc, b))
        except np.linalg.LinAlgError:          # near-converged, badly scaled barrier terms
            lin = lambda b: np.linalg.lstsq(Kmat, b, rcond=None)[0]

        def solve(rc):
            rhs = -rd - G.T @ (W * rp + rc / s)
            dx = lin(rhs)
            dz = W * (G @ dx + rp) + rc / s
            ds = (rc - s * dz) / z
            return dx, dz, ds

        def max_step(v, dv):
            neg = dv < 0
            return min(1.0, np.min(-v[neg] / dv[neg])) if np.any(neg) else 1.0

        dx, dz, ds = solve(-s * z)                          # affine (predictor) step
        a_aff = min(max_step(s, ds), max_step(z, dz))
        mu_aff = (s + a_aff * ds) @ (z + a_aff * dz) / m
        sigma = (mu_aff / mu) ** 3
        dx, dz, ds = solve(-s * z - ds * dz + sigma * mu)   # corrector step
        a = 0.99 * min(max_step(s, ds), max_step(z, dz))
        x, s, z = x + a * dx, s + a * ds, z + a * dz
        s = np.maximum(s, 1e-14 * scale)
        z = np.maximum(z, 1e-14)
    if met is not None:
        return met[0], it, True
    return best[1], it, False


def _polish(H, f, G, h, x, s, z):
    """Active-set polishing of the IPM solution: solve the equality-constrained KKT system for
    the identified active set; keep it only if primal and dual feasible."""
    act = np.where(z > s)[0]
    n = H.shape[0]
    if len(act) == 0:
        xp = np.linalg.solve(H, -f)
        lam = np.zeros(0)
    else:
        Ga = G[act]
        K = np.block([[H, Ga.T], [Ga, np.zeros((len(act), len(act)))]])
        try:
            sol = np.linalg.solve(K, np.concatenate([-f, h[act]]))
        except np.linalg.LinAlgError:
            return x, False
        xp, lam = sol[:n], sol[n:]
    if np.all(G @ xp <= h + 1e-9) and np.all(lam >= -1e-9):
        return xp, True
    return x, False
