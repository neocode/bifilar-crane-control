"""Long-horizon check of the sustained force oscillations found in the Monte Carlo study (RMS force
above 1 kN over the last 10 s of the 30 s runs). The flagged runs are repeated over 300 s to see
whether the oscillation is stationary (a limit cycle) or a slowly decaying transient.
Output: results/limit_cycles.json"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sim import setup, scenarios as SC, simulate as S, params as P

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
T_LONG = 300.0
RMS_FLAG = 1.0          # kN, criterion of Sect. 2.4
WINDOWS = [(20, 30), (90, 100), (190, 200), (290, 300)]


def job(args):
    key, name, sc = args
    sc.T = T_LONG
    log = S.run_nonlinear(setup.build(name), sc)
    t, u, xT, phi = log['t'], log['u'], log['xT'], np.degrees(log['phi'])

    def rms(a, b):
        m = (t >= a) & (t < b)
        return float(np.sqrt(np.mean(u[m] ** 2)))

    last = t >= T_LONG - 50.0
    du = u[last] - np.mean(u[last])
    crossings = np.count_nonzero(np.diff(np.sign(du)) != 0)
    return key, name, dict(
        rms_kN={f'{a}-{b}': rms(a, b) for a, b in WINDOWS},
        period_s=float(2 * 50.0 / crossings) if crossings else float('nan'),
        sat_frac=float(np.mean(np.abs(u[last]) >= 0.999 * P.F_MAX)),
        xT_amp_mm=float(500.0 * (np.max(xT[last]) - np.min(xT[last]))),
        phi_amp_deg=float(0.5 * (np.max(phi[last]) - np.min(phi[last]))),
        tau=float(sc.sample['tau']), mu=float(sc.sample['mu']))


if __name__ == '__main__':
    mc = json.load(open(os.path.join(RES, 'metrics_montecarlo.json')))
    scen = SC.monte_carlo_set()
    jobs = [(k, c, scen[k]) for k in scen for c in setup.ALL
            if np.isfinite(mc[k][c].get('u_rms_res', np.nan)) and mc[k][c]['u_rms_res'] > RMS_FLAG]
    jobs.sort(key=lambda j: 0 if j[1].startswith('MPC') else 1)
    out = {}
    with ProcessPoolExecutor(20) as ex:
        for key, name, r in ex.map(job, jobs, chunksize=1):
            out[f'{key}/{name}'] = r
    ratio = [r['rms_kN']['290-300'] / r['rms_kN']['20-30'] for r in out.values()]
    out['_summary'] = dict(
        runs=len(jobs), stationary=int(sum(abs(q - 1.0) < 0.1 for q in ratio)),
        rms_ratio_300_over_30=[float(min(ratio)), float(max(ratio))] if ratio else None,
        rms_290_300_kN=[float(min(r['rms_kN']['290-300'] for k, r in out.items() if k != '_summary')),
                        float(max(r['rms_kN']['290-300'] for k, r in out.items() if k != '_summary'))] if ratio else None,
        period_s=[float(np.nanmin([r['period_s'] for k, r in out.items() if k != '_summary'])),
                  float(np.nanmax([r['period_s'] for k, r in out.items() if k != '_summary']))] if ratio else None)
    for k, r in out.items():
        if k != '_summary':
            print(k, {w: round(v, 2) for w, v in r['rms_kN'].items()}, 'T=%.2f s sat=%.2f' % (r['period_s'], r['sat_frac']))
    print(out['_summary'])
    json.dump(out, open(os.path.join(RES, 'limit_cycles.json'), 'w'), indent=1)
