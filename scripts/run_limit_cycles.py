"""Long-horizon check of sustained oscillations (results/limit_cycles.json).

Candidates are all runs of the nominal, robustness and Monte Carlo groups whose last 10 s show an RMS
force above 1 kN (force oscillation) or whose swing oscillates with an amplitude above 0.1° over the
last 5 s (half the peak-to-peak value about the mean, so that a static deflection as in S5 is not
counted). Each candidate is repeated over 300 s with the same plant, controller and initial condition
and classified as
  limit_cycle        RMS force above 1 kN over 290-300 s and within 10 % of its 20-30 s value,
  swing_oscillation  swing amplitude above 0.1° over 290-300 s (not a force limit cycle),
  decaying           otherwise.
"""
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
RMS_FLAG = 1.0          # kN
OSC_FLAG = 0.1          # deg
WINDOWS = [(20, 30), (90, 100), (190, 200), (290, 300)]


def job(args):
    group, key, name, sc = args
    sc.T = T_LONG
    log = S.run_nonlinear(setup.build(name), sc)
    t, u, xT, phi = log['t'], log['u'], log['xT'], np.degrees(log['phi'])

    def win(a, b):
        return (t >= a) & (t < b)

    def rms(a, b):
        return float(np.sqrt(np.mean(u[win(a, b)] ** 2)))

    def amp(a, b):
        p = phi[win(a, b)]
        return float(0.5 * (p.max() - p.min()))

    last = t >= T_LONG - 50.0
    du = u[last] - np.mean(u[last])
    dp = phi[last] - np.mean(phi[last])
    cu = np.count_nonzero(np.diff(np.sign(du)) != 0)
    cp = np.count_nonzero(np.diff(np.sign(dp)) != 0)
    r = dict(rms_kN={f'{a}-{b}': rms(a, b) for a, b in WINDOWS},
             phi_amp_deg={f'{a}-{b}': amp(a, b) for a, b in WINDOWS},
             period_s=float(2 * 50.0 / cu) if cu else float('nan'),
             phi_period_s=float(2 * 50.0 / cp) if cp else float('nan'),
             sat_frac=float(np.mean(np.abs(u[last]) >= 0.999 * P.F_MAX)),
             xT_amp_mm=float(500.0 * (np.max(xT[last]) - np.min(xT[last]))))
    if group == 'montecarlo':
        r.update(tau=float(sc.sample['tau']), mu=float(sc.sample['mu']), l=float(sc.sample['l']))
    rr = r['rms_kN']
    if rr['290-300'] > RMS_FLAG and abs(rr['290-300'] / rr['20-30'] - 1.0) < 0.1:
        r['class'] = 'limit_cycle'
    elif r['phi_amp_deg']['290-300'] > OSC_FLAG:
        r['class'] = 'swing_oscillation'
    else:
        r['class'] = 'decaying'
    return group, key, name, r


def candidates():
    sets = {'nominal': SC.nominal_set(), 'robust': SC.robustness_set(), 'montecarlo': SC.monte_carlo_set()}
    jobs = []
    for group, scen in sets.items():
        mets = json.load(open(os.path.join(RES, f'metrics_{group}.json')))
        for k, sc in scen.items():
            for c in setup.ALL:
                m = mets[k][c]
                if 'failed' in m:
                    continue
                if m.get('u_rms_res', 0.0) > RMS_FLAG or m.get('phi_osc', 0.0) > OSC_FLAG:
                    jobs.append((group, k, c, sc))
    return jobs


if __name__ == '__main__':
    jobs = candidates()
    jobs.sort(key=lambda j: 0 if j[2].startswith('MPC') else 1)
    out = {}
    with ProcessPoolExecutor(20) as ex:
        for group, key, name, r in ex.map(job, jobs, chunksize=1):
            out[f'{group}/{key}/{name}'] = r
    summary = {}
    for k, r in out.items():
        group, _, name = k.split('/', 2)
        d = summary.setdefault(group, {}).setdefault(name, {'candidates': 0, 'limit_cycle': 0,
                                                            'swing_oscillation': 0, 'decaying': 0})
        d['candidates'] += 1
        d[r['class']] += 1
    out['_summary'] = summary
    for k, r in sorted(out.items()):
        if k != '_summary':
            print(k, r['class'], {w: round(v, 2) for w, v in r['rms_kN'].items()},
                  {w: round(v, 2) for w, v in r['phi_amp_deg'].items()})
    print(json.dumps(summary, indent=1))
    json.dump(out, open(os.path.join(RES, 'limit_cycles.json'), 'w'), indent=1)
