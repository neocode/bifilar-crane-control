"""Verification of the nonlinear evaluation plant (results/plant_verification.txt)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sim import params as P
from sim import plant

out = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)

prm0 = P.plant_params(mu=0.0, cv=0.0, zeta=0.0)      # conservative plant
s0 = plant.initial_state(prm0)
# --- numerical linearisation at equilibrium --------------------------------------------
n = plant.NSTATE - 1
Jac = np.zeros((n, n)); eps = 1e-6
f0, _, _ = plant.rhs(s0, 0.0, 0.0, 0.0, prm0)
for j in range(n):
    sp = s0.copy(); sp[j] += eps
    fp, _, _ = plant.rhs(sp, 0.0, 0.0, 0.0, prm0)
    sm = s0.copy(); sm[j] -= eps
    fm, _, _ = plant.rhs(sm, 0.0, 0.0, 0.0, prm0)
    Jac[:, j] = (fp[:n] - fm[:n]) / (2 * eps)
ev = np.linalg.eigvals(Jac)
log('Eigenvalues of the linearised multibody plant (12 states, incl. 4 Baumgarte modes):')
for e in sorted(ev, key=lambda z: (round(z.real, 6), z.imag)):
    log(f'   {e.real:+.5f} {e.imag:+.5f}j')
g, l, a, m1, m2 = P.G, P.L_ROPE, P.A_HALF, P.M1, P.M2
w_y, w_th, w_c = np.sqrt(g / l), np.sqrt(3 * g / l), np.sqrt((m1 + m2) * g / (m1 * l))
log(f'Analytical (Eq. 15): w_y={w_y:.5f} (T={2*np.pi/w_y:.3f} s), w_theta=sqrt(3g/l)={w_th:.5f} '
    f'(T={2*np.pi/w_th:.3f} s), w_c={w_c:.5f} (T={2*np.pi/w_c:.3f} s)')
log(f'Old manuscript value w_theta=(a/l)sqrt(2g/l)={a/l*np.sqrt(2*g/l):.5f} (T={2*np.pi/(a/l*np.sqrt(2*g/l)):.3f} s)')

# --- energy conservation and constraint drift, large 3-D free motion -------------------
s = plant.initial_state(prm0, xT=0.0, x_com=0.8, y_com=0.5, theta=np.radians(10))
E0 = plant.total_energy(s, prm0)
t0 = time.time(); Emax = 0.0; cmax = 0.0; Tmin = 1e12; Tmax = 0.0
for k in range(int(30 / P.TS)):
    s, tm, _w = plant.step_zoh(s, 0.0, 0.0, 0.0, prm0, P.TS, P.H_INT)
    Emax = max(Emax, abs(plant.total_energy(s, prm0) - E0))
    r1, r2 = plant.constraint_residual(s, prm0)
    cmax = max(cmax, abs(r1), abs(r2))
log(f'Free 3-D motion (x_CoM=0.8 m, y_CoM=0.5 m, theta=10 deg), 30 s, RK4 h={P.H_INT}s: '
    f'max|E-E0|={Emax:.3e} J (E0 rel {Emax/(m2*g*l):.2e}), max constraint residual={cmax:.2e} m, '
    f'cpu={time.time()-t0:.1f}s')

# --- static rope tension -----------------------------------------------------------------
_, T1, T2 = plant.rhs(plant.initial_state(prm0), 0.0, 0.0, 0.0, prm0)
log(f'Static rope tensions: {T1:.1f} N, {T2:.1f} N (m2 g/2 = {m2*g/2:.1f} N)')

open(os.path.join(os.path.dirname(__file__), '..', 'results', 'plant_verification.txt'), 'w').write('\n'.join(out) + '\n')
