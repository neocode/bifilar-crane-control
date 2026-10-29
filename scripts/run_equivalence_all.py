"""MPC versus the saturated augmented LQR with the MPC weights in all nominal, robustness and Monte Carlo
cases (results/equivalence_all.json). For every case both controllers are run on the nonlinear plant and
the largest difference of the applied commands, the time of the first difference above 1e-6 kN, and the
differences of the settling time and of the mechanical work are recorded."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sim import setup, scenarios as SC, simulate as S, metrics as Mx

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def job(args):
    group, key, sc = args
    a = S.run_nonlinear(setup.build('MPC'), sc)
    b = S.run_nonlinear(setup.build('SatLQR-MPCw'), sc)
    du = np.abs(a['u'] - b['u'])
    ma, mb = Mx.compute(a, sc.xT0), Mx.compute(b, sc.xT0)
    first = np.nonzero(du > 1e-6)[0]
    sat = int(np.sum(np.abs(a['u']) >= setup.P.F_MAX - 1e-9))
    return group, key, dict(max_du=float(du.max()), t_first=float(a['t'][first[0]]) if len(first) else None,
                            t_last=float(a['t'][first[-1]]) if len(first) else None,
                            max_du_start=float(du[a['t'] < 1.0].max()), max_du_later=float(du[a['t'] >= 1.0].max()),
                            dTs=float(abs(ma['Ts'] - mb['Ts'])) if np.isfinite(ma['Ts']) and np.isfinite(mb['Ts'])
                            else (0.0 if ma['Ts'] == mb['Ts'] else float('inf')),
                            dW=float(abs(ma['work'] - mb['work'])), saturated_samples=sat)


if __name__ == '__main__':
    jobs = [('nominal', k, sc) for k, sc in SC.nominal_set().items()]
    jobs += [('robust', k, sc) for k, sc in SC.robustness_set().items()]
    jobs += [('montecarlo', k, sc) for k, sc in SC.monte_carlo_set().items()]
    out = {}
    with ProcessPoolExecutor(16) as ex:
        for group, key, r in ex.map(job, jobs, chunksize=2):
            out.setdefault(group, {})[key] = r
    summ = {}
    for group, d in out.items():
        dus = np.array([r['max_du'] for r in d.values()])
        diff = [k for k, r in d.items() if r['max_du'] > 1e-6]
        summ[group] = dict(n=len(d), max_du_equal=float(max([r['max_du'] for r in d.values() if r['max_du'] <= 1e-6],
                                                          default=0.0)),
                           n_diff=len(diff), max_du=float(dus.max()),
                           t_first_max=float(max([d[k]['t_first'] for k in diff], default=0.0)),
                           t_last_max=float(max([d[k]['t_last'] for k in diff], default=0.0)),
                           dTs_max=float(max([d[k]['dTs'] for k in diff], default=0.0)),
                           later_max=float(max([d[k]['max_du_later'] for k in diff], default=0.0)),
                           later_second=float(sorted([d[k]['max_du_later'] for k in diff])[-2]) if len(diff) > 1 else 0.0,
                           dW_max=float(max([d[k]['dW'] for k in diff], default=0.0)))
    out['_summary'] = summ
    print(json.dumps(summ, indent=1))
    json.dump(out, open(os.path.join(RES, 'equivalence_all.json'), 'w'), indent=1)
