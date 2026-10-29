"""Verification of the IPM QP solver against SciPy BVLS (box QPs) and SLSQP (general QPs)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.optimize import lsq_linear, minimize
from sim.qp import ipm_qp
rng = np.random.default_rng(0)
out = []
emax_box = 0; emax_gen = 0
for trial in range(40):
    n = 30
    M = rng.normal(size=(n, n)); H = M @ M.T + 0.1 * np.eye(n); f = rng.normal(size=n) * 10
    # box QP
    G = np.vstack([np.eye(n), -np.eye(n)]); h = np.ones(2 * n)
    x_ipm, it, ok = ipm_qp(H, f, G, h)
    L = np.linalg.cholesky(H); x_b = lsq_linear(L.T, -np.linalg.solve(L, f), bounds=(-1, 1), method='bvls', tol=1e-14).x
    emax_box = max(emax_box, np.max(np.abs(x_ipm - x_b)))
    # general QP
    A = rng.normal(size=(20, n)); b = rng.uniform(0.5, 2, 20)
    G2 = np.vstack([G, A]); h2 = np.concatenate([h, b])
    x2, it2, ok2 = ipm_qp(H, f, G2, h2)
    cons = [{'type': 'ineq', 'fun': lambda x: h2 - G2 @ x, 'jac': lambda x: -G2}]
    r = minimize(lambda x: 0.5 * x @ H @ x + f @ x, np.zeros(n), jac=lambda x: H @ x + f, constraints=cons,
                 method='SLSQP', options=dict(ftol=1e-14, maxiter=1000))
    emax_gen = max(emax_gen, np.max(np.abs(x2 - r.x)))
s = f'IPM vs BVLS (40 random box QPs, n=30): max |dx| = {emax_box:.2e}\nIPM vs SLSQP (40 random general QPs, n=30, 80 constraints): max |dx| = {emax_gen:.2e}'
print(s)
open(os.path.join(os.path.dirname(__file__), '..', 'results', 'qp_verification.txt'), 'w').write(s + '\n')
