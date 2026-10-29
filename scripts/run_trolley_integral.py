"""PID with integral action in the trolley channel only (K_ia = 0), tuned with the same procedure and
specifications as PID (optimal), and evaluated in S1-S5, R1-R7 and the Monte Carlo study. It tests
the structure suggested in the Discussion (trolley-channel integral action; the swing feedback acts
on the load velocity through the derivative gains). Output: results/trolley_integral.json"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sim import tuning as TU, controllers as C, scenarios as SC, simulate as S, metrics as Mx, setup

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
NAME = 'PID (trolley I)'
SPEC = TU.SPECS['PID (optimal)']
SEEDS = [1, 2, 3, 4]
KEYS = ['KpT', 'KiT', 'KdT', 'Kpa', 'Kia', 'Kda']


def tune(args):
    seed, x0 = args
    tu = TU.Tuner(SPEC, swing_integral=False)
    g, r, c = tu.run(seed=seed, x0=x0)
    return seed, g.tolist(), {k: float(v) for k, v in r.items()}, float(c)


def simulate(args):
    group, key, sc, gains = args
    log = S.run_nonlinear(C.PID(gains, name=NAME), sc)
    m = Mx.compute(log, sc.xT0)
    m['vT_peak'] = float(np.max(np.abs(log['vT'])))
    return group, key, {k: float(v) for k, v in m.items()}


if __name__ == '__main__':
    t0 = time.time()
    G = setup.load_gains()
    lo = np.array([b[0] for b in TU.BOUNDS]); hi = np.array([b[1] for b in TU.BOUNDS])
    starts = [(s, None) for s in SEEDS]
    for i, src in enumerate(('PID (optimal)', 'PD')):        # also start from the related designs
        g = np.array([G[src]['gains'][k] for k in KEYS])
        g[4] = 0.0
        starts.append((10 + i, np.clip(g, lo, hi)))
    with ProcessPoolExecutor(len(starts)) as ex:
        res = list(ex.map(tune, starts))
    seed, gains, design, cost = min(res, key=lambda r: r[3])
    print(f'{NAME}: J={cost:.4f} (seed {seed}) gains={np.round(gains, 3)}')
    print('   ', {k: round(v, 4) for k, v in design.items()})
    jobs = [('nominal', k, sc, gains) for k, sc in SC.nominal_set().items()]
    jobs += [('robust', k, sc, gains) for k, sc in SC.robustness_set().items()]
    mc = SC.monte_carlo_set()
    jobs += [('montecarlo', k, sc, gains) for k, sc in mc.items()]
    out = dict(name=NAME, spec=SPEC, gains=dict(zip(KEYS, gains)), design_metrics=design, cost=cost,
               seed=seed, all_costs=sorted(round(r[3], 6) for r in res), nominal={}, robust={},
               montecarlo={})
    with ProcessPoolExecutor(20) as ex:
        for group, key, m in ex.map(simulate, jobs, chunksize=4):
            out[group][key] = m
    json.dump(out, open(os.path.join(RES, 'trolley_integral.json'), 'w'), indent=1)
    ts = np.array([out['montecarlo'][k]['Ts'] for k in mc])
    print('S1', {k: round(out['nominal']['S1'][k], 3) for k in ('Ts', 'Ts2', 'overshoot', 'phi_peak', 'work')})
    print('S5 final error (mm)', round(out['nominal']['S5']['e_final_mm'], 1))
    print(f'Monte Carlo: settled {np.mean(np.isfinite(ts)) * 100:.0f}%, median t_s {np.median(np.where(np.isfinite(ts), ts, 1e9)):.2f} s')
    print(f'done in {time.time() - t0:.0f} s')
