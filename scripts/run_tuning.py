"""Specification-based tuning of PD and the four PID variants (results/gains.json)."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sim import tuning as TU

SEEDS = [1, 2, 3, 4]


def job(args):
    name, seed, x0 = args
    tu = TU.Tuner(TU.SPECS[name], integral=(name != 'PD'))
    t0 = time.time()
    g, r, c = tu.run(seed=seed, x0=x0)
    return name, seed, g.tolist(), {k: float(v) for k, v in r.items()}, float(c), time.time() - t0


def best_of(results, name):
    cand = [r for r in results if r[0] == name]
    return min(cand, key=lambda r: r[4])


if __name__ == '__main__':
    out = {}
    with ProcessPoolExecutor(20) as ex:
        res1 = list(ex.map(job, [(n, s, None) for n in TU.SPECS for s in SEEDS]))
        # second pass: every PID variant is also started from the best feasible solution of the
        # neighbouring (less/more restrictive) specification sets and of PD, so that the
        # stochastic search cannot return a worse optimum than a nested special case
        seeds_x0 = []
        for n in TU.SPECS:
            if n == 'PD':
                continue
            for m in TU.SPECS:
                g = np.array(best_of(res1, m)[2])
                tu = TU.Tuner(TU.SPECS[n], integral=True)
                lo = np.array([b[0] for b in TU.BOUNDS]); hi = np.array([b[1] for b in TU.BOUNDS])
                seeds_x0.append((n, 10 + list(TU.SPECS).index(m), np.clip(g, lo, hi)))
        res2 = list(ex.map(job, seeds_x0))
    allres = res1 + res2
    for name in TU.SPECS:
        n, seed, g, r, c, dt = best_of(allres, name)
        out[name] = dict(spec=TU.SPECS[name], gains=dict(zip(['KpT', 'KiT', 'KdT', 'Kpa', 'Kia', 'Kda'], g)),
                         design_metrics=r, cost=c, seed=seed,
                         all_costs=sorted([round(x[4], 6) for x in allres if x[0] == name]))
        print(f"{name:14s} J={c:.4f} seed={seed} gains={np.round(g, 3)}")
        print('      ', {k: round(v, 4) for k, v in r.items()})
    out['_settings'] = dict(D_TEST_kN=TU.D_TEST, bounds=TU.BOUNDS, seeds=SEEDS,
                            objective='ITAE(regulation, x_T0=0.5 m) + ITAE(trolley disturbance D_TEST)',
                            optimizer='scipy differential_evolution (sobol init, popsize 50, maxiter 1000) + Nelder-Mead refinement')
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'gains.json'), 'w'), indent=2)
