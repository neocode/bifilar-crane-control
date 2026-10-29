"""Nominal parameters of the trolley - two-rope - beam system and of the evaluation protocol.

Units: SI inside the plant (m, kg, s, N); the control force is expressed in kN in the
controllers and in the linear design model (B scaled by 1000/m1), as in the manuscript.
"""
import numpy as np

# --- mechanical parameters (Stadnik et al. 2022, FME Trans 50:548-560) -----------------
M1 = 500.0      # trolley (traverse) mass, kg
M2 = 4000.0     # beam mass, kg
L_ROPE = 6.0    # rope length, m
A_HALF = 2.2    # half-length of the beam (rope spacing 2a), m
G = 9.81        # m/s^2


def beam_inertia(m2=M2, a=A_HALF):
    """Moment of inertia of a slender uniform beam of length 2a about a transverse
    axis through its centre of mass, J_C = m2 (2a)^2 / 12 = m2 a^2 / 3."""
    return m2 * a * a / 3.0


# --- dissipation and actuator (evaluation plant only; not in the design model) --------
MU_ROLL = 0.008      # specific travel resistance of the trolley (8 N/kN), -
CV_TROLLEY = 100.0   # viscous drive/bearing resistance, N s/m
V_SMOOTH = 0.002     # velocity scale of the tanh-smoothed Coulomb term, m/s
ZETA_SWING = 0.005   # damping ratio of rope terminations / aerodynamic drag, -
TAU_DRIVE = 0.0      # first-order force-actuator lag, s (0 = ideal force source)
BAUMGARTE = 20.0     # Baumgarte constraint stabilisation (alpha = beta), 1/s

# --- control / protocol ------------------------------------------------------------------
F_MAX = 10.0         # actuator force limit, kN
TS = 0.02            # common controller sampling period, s
T_END = 30.0         # unified evaluation horizon, s
H_INT = 0.0005       # RK4 integration step of the plant, s


def plant_params(m1=M1, m2=M2, l=L_ROPE, a=A_HALF, g=G, mu=MU_ROLL, cv=CV_TROLLEY,
                 vs=V_SMOOTH, zeta=ZETA_SWING, tau=TAU_DRIVE, baum=BAUMGARTE):
    """Pack the evaluation-plant parameters into the float64 vector used by plant.py."""
    J = beam_inertia(m2, a)
    Fc = mu * (m1 + m2) * g                       # Coulomb travel resistance, N
    cs = 2.0 * zeta * m2 * np.sqrt(g / l)         # relative-translation damping, N s/m
    omega_th = np.sqrt(m2 * g * a * a / (J * l))  # linear torsional frequency, rad/s
    cth = 2.0 * zeta * J * omega_th               # torsional damping, N m s
    return np.array([m1, m2, l, a, g, J, Fc, cv, vs, cs, cth, tau, baum, baum],
                    dtype=np.float64)


def linear_model(m1=M1, m2=M2, l=L_ROPE, g=G):
    """Continuous-time longitudinal design model x' = A x + B u, u in kN.

    x = [x_T, dx_T/dt, x_CoM, dx_CoM/dt]^T (m, m/s, m, m/s).
    """
    M = m1 + m2
    A = np.array([[0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, m2 * g / (m1 * l), 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, -M * g / (m1 * l), 0.0]])
    B = np.array([[0.0], [1000.0 / m1], [0.0], [-1000.0 / m1]])
    C = np.eye(4)
    Cp = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0 / l, 0.0]])   # regulated outputs: x_T and alpha = x_CoM/l
    return A, B, C, Cp
