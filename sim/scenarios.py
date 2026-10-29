"""Test scenarios (all evaluated over the unified 30 s horizon on the nonlinear plant)."""
import numpy as np

from . import params as P
from .simulate import Scenario


def nominal_set():
    return {
        'S1': Scenario('S1: x_T(0) = 0.5 m (reference case)', xT0=0.5),
        'S2': Scenario('S2: x_T(0) = 5 m (large repositioning)', xT0=5.0),
        'S3': Scenario('S3: initial swing 3 deg, x_T(0) = 0', xT0=0.0, phi0_deg=3.0),
        'S4': Scenario('S4: x_T(0) = 0.5 m, y_CoM(0) = 0.2 m, theta(0) = 5 deg', xT0=0.5, y0=0.2,
                       theta0_deg=5.0),
        'S5': Scenario('S5: 0.5 kN horizontal force on the beam from t = 2 s', xT0=0.0,
                       wind_x=500.0, wind_t=(2.0, 1e9)),
    }


def ramp_set():
    return {'S2r': Scenario('S2 with a 0.5 m/s reference ramp', xT0=5.0, ref_rate=0.5)}


def robustness_set():
    return {
        'R1': Scenario('R1: m2 = 3000 kg', prm=P.plant_params(m2=3000.0)),
        'R2': Scenario('R2: m2 = 5000 kg', prm=P.plant_params(m2=5000.0)),
        'R3': Scenario('R3: l = 4 m', prm=P.plant_params(l=4.0)),
        'R4': Scenario('R4: l = 8 m', prm=P.plant_params(l=8.0)),
        'R5': Scenario('R5: no friction, no damping', prm=P.plant_params(mu=0.0, cv=0.0, zeta=0.0)),
        'R6': Scenario('R6: travel resistance 16 N/kN', prm=P.plant_params(mu=0.016)),
        'R7': Scenario('R7: actuator lag 0.1 s', prm=P.plant_params(tau=0.1)),
    }


MC_RANGES = dict(m2=(3000.0, 5000.0), l=(4.0, 8.0), mu=(0.0, 0.016), zeta=(0.0, 0.02),
                 tau=(0.0, 0.1), xT0=(0.25, 2.0), phi0_deg=(-3.0, 3.0))


def monte_carlo_set(n=200, seed=2026):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        d = {k: rng.uniform(*v) for k, v in MC_RANGES.items()}
        prm = P.plant_params(m2=d['m2'], l=d['l'], mu=d['mu'], zeta=d['zeta'], tau=d['tau'])
        sc = Scenario(f'MC{i:03d}', xT0=d['xT0'], phi0_deg=d['phi0_deg'], prm=prm)
        sc.sample = d
        out[f'MC{i:03d}'] = sc
    return out
