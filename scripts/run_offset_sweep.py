"""Initial-offset sweep on the nominal plant (integrator saturation of PID (optimal)) and rank
correlations of the Monte Carlo settling times with the sampled quantities.
Outputs: results/offset_sweep.json, results/mc_correlations.json"""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.stats import spearmanr
from sim import setup, simulate as S, metrics as Mx

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
out = {}
for x0 in (0.5, 1.0, 1.5, 2.0):
    for name in ('PD', 'PID (optimal)', 'LQR', 'MPC'):
        c = setup.build(name)
        lg = S.run_nonlinear(c, S.Scenario('sweep', xT0=x0))
        m = Mx.compute(lg, x0)
        r = dict(Ts=m['Ts'], e_final_mm=m['e_final_mm'], phi_peak=m['phi_peak'])
        if name.startswith('PID'):
            r.update(KiT_xiT=c.KiT * c.xiT, Kia_xia=c.Kia * c.xia)
        out[f'{x0}/{name}'] = r
        print(x0, name, {k: round(v, 3) for k, v in r.items()})
json.dump(out, open(os.path.join(RES, 'offset_sweep.json'), 'w'), indent=1)

mc = json.load(open(os.path.join(RES, 'metrics_montecarlo.json')))
keys = list(mc)
cor = {}
for name in setup.ALL:
    ts = np.array([mc[k][name]['Ts'] for k in keys])
    ts = np.where(np.isfinite(ts), ts, 40.0)
    cor[name] = {p: float(spearmanr([mc[k]['_sample'][p] for k in keys], ts)[0])
                 for p in ('m2', 'l', 'mu', 'zeta', 'tau', 'xT0', 'phi0_deg')}
    cor[name]['abs_l_minus_6'] = float(spearmanr([abs(mc[k]['_sample']['l'] - 6) for k in keys], ts)[0])
json.dump(cor, open(os.path.join(RES, 'mc_correlations.json'), 'w'), indent=1)
print({k: round(max(abs(v) for v in d.values()), 2) for k, d in cor.items()})
