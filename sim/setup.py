"""Construction of all controllers used in the study (single source of truth)."""
import json
import os
import numpy as np

from . import params as P
from . import controllers as C
from .linear import c2d, dlqr
from .terminal import TerminalSet

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

# LQR weights (manuscript, Sect. 2.2.3): Q = diag(100, 10, 5000, 100), R = 0.01
Q_LQR = np.diag([100.0, 10.0, 5000.0, 100.0])
R_LQR = 0.01
# MPC weights (manuscript, Table 4): Q = diag(1000, 10, 5000, 100), R = 0.1, R_du = 0.5
Q_MPC = np.diag([1000.0, 10.0, 5000.0, 100.0])
R_MPC = 0.1
RDU_MPC = 0.5
N_MPC = 100                      # 2.0 s prediction horizon at TS = 0.02 s (all moves free)
ALPHA_MAX = np.radians(2.0)      # state-constrained MPC: swing-angle limit
V_MAX = 0.5                      # state-constrained MPC: trolley speed limit, m/s

CLASSICAL = ['PD', 'PID (soft)', 'PID (medium)', 'PID (optimal)', 'PID (hard)', 'LQR']
ALL = CLASSICAL + ['MPC', 'MPC-SC']


def design_model(Ts=P.TS):
    A, B, Cm, Cp = P.linear_model()
    Ad, Bd = c2d(A, B, Ts)
    return A, B, Ad, Bd


def load_gains():
    return json.load(open(os.path.join(RESULTS, 'gains.json')))


def lqr_gain(Ts=P.TS):
    _, _, Ad, Bd = design_model(Ts)
    K, Pm, ev = dlqr(Ad, Bd, Q_LQR, np.array([[R_LQR]]))
    return K, Pm, ev


_TERM_CACHE = {}


def terminal_sets():
    """O_inf of the MPC terminal law for the input constraint only and for input + state limits."""
    if not _TERM_CACHE:
        _, _, Ad, Bd = design_model()
        Pa, Ka, Aa, Ba = C.mpc_terminal_ingredients(Ad, Bd, Q_MPC, R_MPC, RDU_MPC)
        Acl = Aa - Ba @ Ka
        Hc = np.vstack([-Ka, Ka])
        hc = np.array([P.F_MAX, P.F_MAX])
        _TERM_CACHE['input'] = TerminalSet(Acl, Hc, hc)
        e_x, e_v = np.zeros((1, 5)), np.zeros((1, 5))
        e_x[0, 2], e_v[0, 1] = 1.0, 1.0
        xl = P.L_ROPE * np.sin(ALPHA_MAX)
        Hc2 = np.vstack([-Ka, Ka, e_x, -e_x, e_v, -e_v])
        hc2 = np.array([P.F_MAX, P.F_MAX, xl, xl, V_MAX, V_MAX])
        _TERM_CACHE['state'] = TerminalSet(Acl, Hc2, hc2)
        _TERM_CACHE['Ka'] = Ka
    return _TERM_CACHE


def build(name, gains=None, ki_scale=(1.0, 1.0)):
    """Return a fresh controller instance by name."""
    if name in ('PD', 'PID (soft)', 'PID (optimal)', 'PID (medium)', 'PID (hard)'):
        g = (gains or load_gains())[name]['gains']
        vec = [g['KpT'], g['KiT'] * ki_scale[0], g['KdT'], g['Kpa'], g['Kia'] * ki_scale[1], g['Kda']]
        return C.PID(vec, name=name)
    if name == 'LQR':
        K, _, _ = lqr_gain()
        return C.StateFeedback(K, name='LQR')
    _, _, Ad, Bd = design_model()
    ts = terminal_sets()
    if name == 'MPC':
        return C.MPC(Ad, Bd, Q_MPC, R_MPC, RDU_MPC, N_MPC, name='MPC', term_set=ts['input'])
    if name == 'MPC-SC':
        return C.MPC(Ad, Bd, Q_MPC, R_MPC, RDU_MPC, N_MPC, name='MPC-SC', alpha_max=ALPHA_MAX,
                     v_max=V_MAX, term_set=ts['state'])
    if name == 'SatLQR-MPCw':
        return C.AugmentedLQR(ts['Ka'], name='SatLQR-MPCw')
    raise KeyError(name)
