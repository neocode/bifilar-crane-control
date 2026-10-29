"""Sensitivity of the S1 results of PID (optimal) to the saturation threshold of the conditional integration.

The force budget of PID (optimal) equals F_max, so its first command in S1 lies 1e-9 kN below the threshold
F_max + SAT_TOL (Sect. 2.2.2). The script repeats S1 with K_pT raised just below and just above the value at
which the first command exceeds the threshold (results/saturation_threshold.json)."""
import os as _os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_v, '1')
import sys, os, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sim import params as P, setup, scenarios as SC, simulate as S, metrics as Mx

X0 = 0.5
KPT_SAT = (P.F_MAX + P.SAT_TOL) / X0          # first command |K_pT x_T(0)| reaches the threshold


def job(kpt):
    g = copy.deepcopy(setup.load_gains())
    g['PID (optimal)']['gains']['KpT'] = kpt
    m = Mx.compute(S.run_nonlinear(setup.build('PID (optimal)', gains=g), SC.nominal_set()['S1']), X0)
    return {k: float(m[k]) for k in ('Ts', 'Ts2', 'overshoot', 'phi_peak', 'IAE', 'ITAE', 'work')}


if __name__ == '__main__':
    kpt0 = setup.load_gains()['PID (optimal)']['gains']['KpT']
    out = dict(KpT_tuned=kpt0, KpT_threshold=KPT_SAT, margin_kN=P.F_MAX + P.SAT_TOL - X0 * kpt0,
               tuned=job(kpt0), below=job(KPT_SAT - 1e-10), above=job(KPT_SAT + 1e-10))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'saturation_threshold.json'), 'w'),
              indent=1)
