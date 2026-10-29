"""Performance metrics, identical for every controller and scenario (30 s horizon)."""
import numpy as np

PHI_BAND_DEG = 0.1      # residual-swing band used for the swing settling time, deg
BAND_MIN = 0.025        # minimum trolley-position band, m
PHI_LIMIT_DEG = 2.0     # MPC-SC swing limit (reported for every controller), deg
V_LIMIT = 0.5           # MPC-SC trolley-speed limit, m/s
LIMIT_TOL = 1e-4        # tolerance for counting a limit as exceeded (deg or m/s)


def settling_time(t, y, band):
    """Earliest time after which |y| stays within the band until the end of the horizon."""
    out = np.where(np.abs(y) > band)[0]
    if len(out) == 0:
        return 0.0
    if out[-1] == len(y) - 1:
        return np.inf
    return t[out[-1] + 1]


def compute(log, xT0):
    t, xT, phi, u = log['t'], log['xT'], np.degrees(log['phi']), log['u']
    band = max(0.05 * abs(xT0), BAND_MIN)
    m = {}
    m['Ts'] = settling_time(t, xT, band)
    m['band2'] = max(0.02 * abs(xT0), 0.01)                  # 2 % band (minimum 10 mm)
    m['Ts2'] = settling_time(t, xT, m['band2'])
    m['Ts_phi'] = settling_time(t, phi, PHI_BAND_DEG)
    m['overshoot'] = (max(0.0, -np.min(np.sign(xT0) * xT)) / abs(xT0) * 100.0   # excursion beyond the target
                      if xT0 != 0 else np.nan)
    m['Fpeak'] = np.max(np.abs(u))
    m['work'] = log['work'] / 1000.0                       # kJ
    m['phi_peak'] = np.max(np.abs(phi))
    m['phi_res'] = np.max(np.abs(phi[t >= t[-1] - 5.0]))   # max |phi| over last 5 s
    last = phi[t >= t[-1] - 5.0]
    m['phi_osc'] = 0.5 * (np.max(last) - np.min(last))     # swing oscillation amplitude over last 5 s
    m['IAE'] = np.trapezoid(np.abs(xT), t)
    m['ITAE'] = np.trapezoid(t * np.abs(xT), t)
    m['Trope'] = np.nanmax(log['Trope']) / 1000.0 if np.any(np.isfinite(log['Trope'])) else np.nan
    m['e_final_mm'] = abs(xT[-1]) * 1000.0
    m['u_rms_res'] = np.sqrt(np.mean(u[t >= t[-1] - 10.0] ** 2))  # residual activity 20-30 s
    m['band'] = band
    # time spent above the MPC-SC limits (sampled at the controller period) and largest excess
    dt = t[1] - t[0]
    v = np.abs(log['vT'])
    m['t_over_phi'] = float(np.sum(np.abs(phi) > PHI_LIMIT_DEG + LIMIT_TOL) * dt)
    m['t_over_v'] = float(np.sum(v > V_LIMIT + LIMIT_TOL) * dt)
    m['phi_excess'] = float(max(0.0, np.max(np.abs(phi)) - PHI_LIMIT_DEG))
    m['v_excess'] = float(max(0.0, np.max(v) - V_LIMIT))
    return m
