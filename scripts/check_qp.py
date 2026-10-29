"""Verification of the IPM QP solver (results/qp_verification.txt).

1. Random problems: IPM against SciPy BVLS (box QPs) and SLSQP (general QPs, ftol = 1e-10, the
   tightest tolerance at which SLSQP reports success on all problems); every solver must report
   success, and the IPM solutions are certified by their KKT residuals.
2. Real-size problems: all QPs of the state-constrained MPC (MPC-SC) solved during the nominal
   scenarios S1-S3 (102 variables, 602 inequality constraints); for each solution the KKT conditions
   are evaluated with a strict, per-constraint activity test: primal feasibility, stationarity with
   non-negative multipliers on the active set (computed by NNLS) and complementarity, which for a
   convex QP certifies optimality. A negative test checks that slightly perturbed plans are rejected.
"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.optimize import lsq_linear, minimize, nnls
from sim.qp import ipm_qp
from sim import setup, scenarios as SC, simulate as S


def kkt(H, f, G, h, x):
    """KKT residuals of x for min 0.5 x'Hx + f'x s.t. Gx <= h. A constraint counts as active only if its
    own gap is below 1e-9 (1 + |h_i|); the multipliers of the active constraints are the non-negative
    least-squares solution of the stationarity condition (dual feasibility). Returned: relative primal
    violation, per-component relative stationarity residual, complementarity max(lambda_i |gap_i|)
    relative to the gradient scale, and the number of active constraints."""
    gap = G @ x - h
    viol = max(0.0, float(np.max(gap / (1.0 + np.abs(h)))))
    act = np.where(gap >= -1e-9 * (1.0 + np.abs(h)))[0]
    grad = H @ x + f
    lam = np.zeros(len(h))
    if len(act):
        lam[act], _ = nnls(G[act].T, -grad, maxiter=50 * len(act))
    r = grad + G.T @ lam
    stat = float(np.max(np.abs(r) / (1.0 + np.abs(H @ x) + np.abs(f))))
    comp = float(np.max(lam * np.abs(gap)) / (1.0 + np.max(np.abs(grad))))
    return viol, stat, comp, len(act)


rng = np.random.default_rng(0)
emax_box = 0; emax_gen = 0; fails = []; kkt_rand = []
for trial in range(40):
    n = 30
    M = rng.normal(size=(n, n)); H = M @ M.T + 0.1 * np.eye(n); f = rng.normal(size=n) * 10
    # box QP
    G = np.vstack([np.eye(n), -np.eye(n)]); h = np.ones(2 * n)
    x_ipm, it, ok = ipm_qp(H, f, G, h)
    L = np.linalg.cholesky(H)
    rb = lsq_linear(L.T, -np.linalg.solve(L, f), bounds=(-1, 1), method='bvls', tol=1e-14)
    if not ok or not rb.success:
        fails.append(('box', trial, ok, rb.success))
    emax_box = max(emax_box, np.max(np.abs(x_ipm - rb.x)))
    # general QP
    A = rng.normal(size=(20, n)); b = rng.uniform(0.5, 2, 20)
    G2 = np.vstack([G, A]); h2 = np.concatenate([h, b])
    x2, it2, ok2 = ipm_qp(H, f, G2, h2)
    cons = [{'type': 'ineq', 'fun': lambda x: h2 - G2 @ x, 'jac': lambda x: -G2}]
    r = minimize(lambda x: 0.5 * x @ H @ x + f @ x, np.zeros(n), jac=lambda x: H @ x + f, constraints=cons,
                 method='SLSQP', options=dict(ftol=1e-10, maxiter=1000))
    if not ok2 or not r.success:
        fails.append(('general', trial, ok2, r.success))
    emax_gen = max(emax_gen, np.max(np.abs(x2 - r.x)))
    kkt_rand += [kkt(H, f, G, h, x_ipm), kkt(H, f, G2, h2, x2)]
kr = np.array(kkt_rand)
out = [f'IPM vs BVLS (40 random box QPs, n=30): max |dx| = {emax_box:.2e}',
       f'IPM vs SLSQP (40 random general QPs, n=30, 80 constraints): max |dx| = {emax_gen:.2e}',
       f'Random problems: all solvers reported success: {not fails} {fails if fails else ""}'.rstrip(),
       f'Random problems, IPM KKT residuals (relative): max primal violation = {kr[:, 0].max():.1e}, '
       f'max stationarity = {kr[:, 1].max():.1e}, max complementarity = {kr[:, 2].max():.1e}']

records = []
for key in ('S1', 'S2', 'S3'):
    c = setup.build('MPC-SC')
    solver = c._ipm

    def recording(H, f, G, h, x0=None, _solver=solver, _key=key):
        x, it, ok = _solver(H, f, G, h, x0=x0)
        records.append((_key, H, f, G, h, x, ok, x0 is not None))
        return x, it, ok

    c._ipm = recording
    S.run_nonlinear(c, SC.nominal_set()[key])
first = [r for r in records if not r[7]]
restarts = [r for r in records if r[7]]
res = np.array([kkt(H, f, G, h, x) for (_, H, f, G, h, x, ok, _r) in records if ok])
# negative test: perturbing the inputs of the optimal plans by 0.1 % must make every certificate fail
prng = np.random.default_rng(1)
passed_bad = 0
for (_, H, f, G, h, x, ok, _r) in records[::45]:
    xb = x.copy()
    xb[:-2] = xb[:-2] * (1.0 + 1e-3 * prng.standard_normal(len(xb) - 2))
    v, st, cp, na = kkt(H, f, G, h, xb)
    passed_bad += (v < 1e-9 and st < 1e-6)
n_var, n_con = records[0][1].shape[0], records[0][3].shape[0]
out.append(f'MPC-SC QPs of S1-S3 ({len(first)} problems, {n_var} variables, {n_con} constraints): '
           f'converged at first attempt {sum(r[6] for r in first)}, restarts {len(restarts)}, '
           f'converged after restart {sum(r[6] for r in restarts)}')
out.append(f'MPC-SC KKT residuals (relative): max primal violation = {res[:, 0].max():.1e}, '
           f'max stationarity = {res[:, 1].max():.1e}, max complementarity = {res[:, 2].max():.1e}, '
           f'active constraints per QP up to {int(res[:, 3].max())}')
out.append(f'Negative test: perturbed plans (0.1 %) accepted by the certificate: {passed_bad} of '
           f'{len(records[::45])}')
print('\n'.join(out))
open(os.path.join(os.path.dirname(__file__), '..', 'results', 'qp_verification.txt'), 'w').write('\n'.join(out) + '\n')
