"""All closed-loop simulations of the revised manuscript.

Groups (results/metrics_<group>.json, time series in results/runs/<group>/):
  nominal     : scenarios S1-S5, all controllers, nonlinear plant
  robust      : parameter / dissipation / actuator variations R1-R7 (x_T(0) = 0.5 m)
  linear      : scenario S1 on the linear design model (linearisation check)
  equivalence : MPC versus saturated LQR with the MPC weights (S1, S2, S3)
  ablation    : PID (optimal) with scaled integral gains (S1 and S5)
  montecarlo  : 200 random plants / initial conditions
Usage: python run_experiments.py [group ...]
"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from sim import setup, scenarios as SC, simulate as S, metrics as Mx

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
GROUPS = ['nominal', 'robust', 'linear', 'equivalence', 'ablation', 'ramp', 'montecarlo']


def task(args):
    group, sc_key, ctrl_name, sc, opts = args
    ctrl = setup.build(ctrl_name, ki_scale=opts.get('ki_scale', (1.0, 1.0)))
    t0 = time.time()
    label = opts.get('label', ctrl_name)
    try:
        if opts.get('linear'):
            log = S.run_linear(ctrl, sc)
        else:
            log = S.run_nonlinear(ctrl, sc)
    except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as e:
        return group, sc_key, label, {'failed': str(e)}
    cpu = time.time() - t0
    m = Mx.compute(log, sc.xT0)
    m['cpu_s'] = cpu
    if hasattr(ctrl, 'solve_times') and len(ctrl.solve_times):
        st = np.array(ctrl.solve_times) * 1e3
        m['qp_mean_ms'], m['qp_max_ms'], m['qp_p99_ms'] = float(st.mean()), float(st.max()), float(np.percentile(st, 99))
        m['qp_iter_mean'] = float(np.mean(ctrl.iters))
        if len(ctrl.term_ok):
            m['term_ok_frac'] = float(np.mean(ctrl.term_ok))
        if len(ctrl.slack):
            m['slack_max'] = float(np.nanmax(np.array(ctrl.slack)))
            m['fallbacks'] = int(ctrl.fallbacks)
    m['vT_peak'] = float(np.max(np.abs(log['vT'])))
    m['yc_peak'] = float(np.max(np.abs(log['yc'])))
    m['theta_peak_deg'] = float(np.degrees(np.max(np.abs(log['theta']))))
    if 'constraint_residual' in log:
        m['constraint_residual'] = float(log['constraint_residual'])
    if opts.get('save', True):
        d = os.path.join(RES, 'runs', group, sc_key)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, label.replace(' ', '_').replace('(', '').replace(')', '') + '.npz'),
                            **{k: v for k, v in log.items() if isinstance(v, np.ndarray)})
    return group, sc_key, label, {k: (float(v) if np.isscalar(v) else v) for k, v in m.items()}


def jobs_for(group):
    J = []
    if group == 'nominal':
        for k, sc in SC.nominal_set().items():
            for c in setup.ALL:
                J.append((group, k, c, sc, {}))
    elif group == 'robust':
        for k, sc in SC.robustness_set().items():
            for c in setup.ALL:
                J.append((group, k, c, sc, {}))
    elif group == 'linear':
        sc = SC.nominal_set()['S1']
        for c in setup.ALL:
            J.append((group, 'S1', c, sc, {'linear': True}))
        sc2 = SC.nominal_set()['S2']
        for c in setup.ALL:
            J.append((group, 'S2', c, sc2, {'linear': True}))
    elif group == 'equivalence':
        for k in ('S1', 'S2', 'S3', 'S5'):
            sc = SC.nominal_set()[k]
            J.append((group, k, 'SatLQR-MPCw', sc, {}))
            J.append((group, k, 'MPC', sc, {}))
    elif group == 'ablation':
        for k in ('S1', 'S5'):
            sc = SC.nominal_set()[k] if k == 'S1' else SC.nominal_set()['S5']
            for cT, ca in [(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1, 1), (1.5, 1.5), (2, 2)]:
                J.append((group, k, 'PID (optimal)', sc, {'ki_scale': (cT, ca), 'label': f'KiT x{cT}, Kia x{ca}'}))
        sc = SC.robustness_set()['R6']
        for cT, ca in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            J.append((group, 'R6', 'PID (optimal)', sc, {'ki_scale': (cT, ca), 'label': f'KiT x{cT}, Kia x{ca}'}))
    elif group == 'ramp':
        sc = SC.ramp_set()['S2r']
        for c in ('PD', 'PID (optimal)', 'PID (hard)', 'LQR', 'MPC'):
            J.append((group, 'S2r', c, sc, {}))
    elif group == 'montecarlo':
        for k, sc in SC.monte_carlo_set().items():
            for c in setup.ALL:
                J.append((group, k, c, sc, {'save': False}))
    return J


if __name__ == '__main__':
    groups = sys.argv[1:] or GROUPS
    for g in groups:
        t0 = time.time()
        J = jobs_for(g)
        J.sort(key=lambda j: 0 if j[2].startswith('MPC') else 1)    # long jobs first
        out = {}
        with ProcessPoolExecutor(20) as ex:
            for group, k, label, m in ex.map(task, J, chunksize=1):
                out.setdefault(k, {})[label] = m
        if g == 'montecarlo':
            for k, sc in SC.monte_carlo_set().items():
                out[k]['_sample'] = sc.sample
        json.dump(out, open(os.path.join(RES, f'metrics_{g}.json'), 'w'), indent=1)
        print(f'{g}: {len(J)} runs in {time.time() - t0:.0f} s')
