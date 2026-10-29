"""Print the summary numbers (gains, eigenvalues, statistics, checks) collected from ../results."""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.join(HERE, '..')
sys.path.insert(0, REV)
RES = os.path.join(REV, 'results')


def J(name):
    return json.load(open(os.path.join(RES, name)))


def run(group, sc, name):
    f = os.path.join(RES, 'runs', group, sc, name.replace(' ', '_').replace('(', '').replace(')', '') + '.npz')
    return dict(np.load(f))


def pctl(v, q):
    v = np.asarray(v, float)
    r = np.percentile(np.where(np.isfinite(v), v, 1e9), q)
    return float(np.inf) if r > 1e8 else float(r)


def collect():
    from sim import setup, params as P
    from sim.linear import z_to_s
    R = {}
    G = J('gains.json')
    R['gains'] = {k: v['gains'] for k, v in G.items() if not k.startswith('_')}
    R['specs'] = {k: v['spec'] for k, v in G.items() if not k.startswith('_')}
    R['design'] = {k: v['design_metrics'] for k, v in G.items() if not k.startswith('_')}
    R['tune'] = {k: dict(cost=v['cost'], all_costs=v['all_costs'], itae_r=v['design_metrics']['ITAE'],
                         itae_d=v['design_metrics']['ITAE_d']) for k, v in G.items() if not k.startswith('_')}
    R['satthr'] = J('saturation_threshold.json')
    K, Pm, evz = setup.lqr_gain()
    R['K_lqr'] = K.ravel().tolist()
    evs = z_to_s(evz, P.TS)
    i_fast = np.argmin(evs.real)
    R['lqr_zfast'] = float(np.real(evz[i_fast]))
    R['lqr_sfast'] = float(evs[i_fast].real)
    reals = sorted([e for e in evs if abs(e.imag) < 1e-9 and e.real > -50], key=lambda e: e.real)
    R['lqr_s2'] = float(reals[0].real)
    cplx = [e for e in evs if e.imag > 1e-9][0]
    R['lqr_s3r'], R['lqr_s3i'] = float(cplx.real), float(cplx.imag)
    R['lqr_poles_s'] = evs
    ts = setup.terminal_sets()
    R['tstar_input'] = ts['input'].tstar
    R['tstar_state'] = ts['state'].tstar
    # plant verification
    s = open(os.path.join(RES, 'plant_verification.txt')).read()
    R['energy_err'] = float(re.search(r'E0 rel ([0-9.e+-]+)\)', s).group(1))
    R['constraint_err'] = float(re.search(r'max constraint residual=([0-9.e+-]+) m', s).group(1))
    s = open(os.path.join(RES, 'integration_check.txt')).read()
    lines = [l for l in s.splitlines() if 'h=0.50 ms' in l]
    R['int_err_x'] = max(float(re.search(r'dx_T\|=([0-9.e+-]+)', l).group(1)) for l in lines)
    R['int_err_phi'] = max(float(re.search(r'dphi\|=([0-9.e+-]+)', l).group(1)) for l in lines)
    qv = open(os.path.join(RES, 'qp_verification.txt')).read()
    R['qp_box_err'] = float(re.search(r'BVLS .*?= ([0-9.e+-]+)', qv).group(1))
    R['qp_gen_err'] = float(re.search(r'SLSQP .*?= ([0-9.e+-]+)', qv).group(1))
    hc = J('horizon_check.json')
    R['N50_diff'] = hc['max_du_N50_vs_N100']
    # metrics
    R['nom'] = J('metrics_nominal.json')
    R['rob'] = J('metrics_robust.json')
    R['lin'] = J('metrics_linear.json')
    R['abl'] = J('metrics_ablation.json')
    R['eqv'] = J('metrics_equivalence.json')
    R['mc'] = J('metrics_montecarlo.json')
    diffs = []
    for sc in R['eqv']:
        a = run('equivalence', sc, 'MPC')['u']
        b = run('equivalence', sc, 'SatLQR-MPCw')['u']
        diffs.append(float(np.max(np.abs(a - b))))
    R['equiv_diff'] = max(max(diffs), 1e-12)
    qt = J('qp_timing.json')
    R['qp_mpc_mean'], R['qp_mpc_p99'], R['qp_mpc_max'] = qt['MPC']['mean_ms'], qt['MPC']['p99_ms'], qt['MPC']['max_ms']
    R['qp_sc_mean'], R['qp_sc_p99'], R['qp_sc_max'] = qt['MPC-SC']['mean_ms'], qt['MPC-SC']['p99_ms'], qt['MPC-SC']['max_ms']
    R['term_mpc_min'] = min(qt[f'MPC/{k}']['term_ok_frac'] for k in ('S1', 'S2', 'S3', 'S4', 'S5'))
    R['term_sc'] = {k: qt[f'MPC-SC/{k}'] for k in ('S1', 'S2', 'S3', 'S4', 'S5')}
    # Monte Carlo summary
    mcs = {}
    for c in setup.ALL:
        rows = [R['mc'][k][c] for k in R['mc']]
        ok = [r for r in rows if 'failed' not in r]
        ts = np.array([r['Ts'] for r in ok])
        ts_inf = np.where(np.isfinite(ts), ts, np.inf)
        mcs[c] = dict(n=len(rows), failed=len(rows) - len(ok),
                      settled=float(np.sum(np.isfinite(ts_inf)) / len(rows)),
                      ts_med=pctl(ts_inf, 50), ts_p90=pctl(ts_inf, 90),
                      phi_med=float(np.median([r['phi_peak'] for r in ok])),
                      phi_p95=float(np.percentile([r['phi_peak'] for r in ok], 95)),
                      w_med=float(np.median([r['work'] for r in ok])),
                      e_med=float(np.median([r['e_final_mm'] for r in ok])),
                      e_p95=float(np.percentile([r['e_final_mm'] for r in ok], 95)),
                      vmax_p95=float(np.percentile([r['vT_peak'] for r in ok], 95)),
                      fallbacks=int(sum(r.get('fallbacks', 0) for r in ok)))
    # sustained oscillations (results/limit_cycles.json): runs flagged at 30 s (RMS force above 1 kN or swing
    # oscillation above 0.1 deg over the last seconds) and classified after a 300 s continuation
    import math
    LL = J('limit_cycles.json')
    cls = {k: r['class'] for k, r in LL.items() if k != '_summary'}

    def cl(group, key, c):
        return cls.get(f'{group}/{key}/{c}', 'none')

    mc_keys = [k for k in R['mc'] if k.startswith('MC')]
    lc, sw = {}, {}
    for c in setup.ALL:
        lc[c] = [k for k in mc_keys if cl('montecarlo', k, c) == 'limit_cycle']
        sw[c] = [k for k in mc_keys if cl('montecarlo', k, c) == 'swing_oscillation']
        n = len(mc_keys)
        mcs[c]['lc'] = len(lc[c]) / n
        mcs[c]['swosc'] = len(sw[c]) / n
        mcs[c]['sustained'] = (len(lc[c]) + len(sw[c])) / n
        mcs[c]['converged'] = sum(1 for k in mc_keys if 'failed' not in R['mc'][k][c]
                                  and np.isfinite(R['mc'][k][c]['Ts'])
                                  and cl('montecarlo', k, c) in ('none', 'decaying')) / n
        mcs[c]['settled_osc'] = sum(1 for k in lc[c] + sw[c] if np.isfinite(R['mc'][k][c]['Ts']))
    force_flag = {c: [k for k in mc_keys if R['mc'][k][c].get('u_rms_res', 0) > 1.0] for c in setup.ALL}
    trans = [(k, c) for c in setup.ALL for k in force_flag[c] if cl('montecarlo', k, c) != 'limit_cycle']
    allk = sorted(set(k for c in lc for k in lc[c]))
    st = [LL[f'montecarlo/{k}/{c}'] for c in lc for k in lc[c]]
    R['lc'] = dict(n={c: len(lc[c]) for c in lc}, n_sw={c: len(sw[c]) for c in sw}, transient=trans,
                   tau_min=math.floor(1000 * min(R['mc'][k]['_sample']['tau'] for k in allk)) / 1000,
                   mu_max=max(R['mc'][k]['_sample']['mu'] for k in allk),
                   rms_min=min(R['mc'][k][c]['u_rms_res'] for c in lc for k in lc[c]),
                   rms_max=max(R['mc'][k][c]['u_rms_res'] for c in lc for k in lc[c]),
                   period=(min(r['period_s'] for r in st), max(r['period_s'] for r in st)),
                   sat=(min(r['sat_frac'] for r in st), max(r['sat_frac'] for r in st)),
                   drift=max(abs(r['rms_kN']['290-300'] / r['rms_kN']['20-30'] - 1.0) for r in st),
                   tau_min_by={c: (min(R['mc'][k]['_sample']['tau'] for k in lc[c]) if lc[c] else None)
                               for c in setup.ALL})
    # swing oscillations of the low-gain PID sets: rope lengths and amplitudes
    swl = [(k, c) for c in sw for k in sw[c]]
    R['sw'] = dict(l_min={c: (min(R['mc'][k]['_sample']['l'] for k in sw[c]) if sw[c] else None) for c in setup.ALL},
                   amp=(min(LL[f'montecarlo/{k}/{c}']['phi_amp_deg']['290-300'] for k, c in swl) if swl else 0.0,
                        max(LL[f'montecarlo/{k}/{c}']['phi_amp_deg']['290-300'] for k, c in swl) if swl else 0.0),
                   period=(min(LL[f'montecarlo/{k}/{c}']['phi_period_s'] for k, c in swl) if swl else 0.0,
                           max(LL[f'montecarlo/{k}/{c}']['phi_period_s'] for k, c in swl) if swl else 0.0))
    # samples beyond the linear stability limits (rope length; drive lag with low friction)
    MGJ = J('linear_margins.json')
    R['beyond'] = {}
    for c in ('PID (soft)', 'PID (medium)', 'PID (optimal)'):
        lim = MGJ[c]['l_unstable_from']
        ks = [k for k in mc_keys if R['mc'][k]['_sample']['l'] >= lim]
        R['beyond'][c] = dict(n=len(ks), n_osc=sum(1 for k in ks if cl('montecarlo', k, c) == 'swing_oscillation'))
    mu_lim = math.ceil(1000 * R['lc']['mu_max'] - 1e-9) / 1000   # the bound stated in the text (N/kN, rounded up)
    for c in ('PID (hard)', 'LQR', 'MPC', 'MPC-SC'):
        tc = MGJ['MPC' if c == 'MPC-SC' else c]['tau_crit']
        ks = [k for k in mc_keys if R['mc'][k]['_sample']['tau'] > tc and R['mc'][k]['_sample']['mu'] <= mu_lim + 1e-12]
        R['beyond'][c] = dict(n=len(ks), n_lc=sum(1 for k in ks if cl('montecarlo', k, c) == 'limit_cycle'))
    # sustained oscillations in the nominal and robustness cases
    R['osc_cases'] = {k: v for k, v in cls.items() if not k.startswith('montecarlo') and
                      v in ('limit_cycle', 'swing_oscillation')}
    R['osc_amp'] = {k: LL[k]['phi_amp_deg']['290-300'] for k in R['osc_cases']}
    R['osc_period'] = {k: LL[k]['phi_period_s'] for k in R['osc_cases']}
    R['mcs'] = mcs
    R['ramp'] = J('metrics_ramp.json')['S2r']
    R['offset'] = J('offset_sweep.json')
    R['sens'] = J('pid_sensitivity.json')
    R['mccor'] = J('mc_correlations.json')
    lin = J('metrics_linear.json')
    R['lin_term_mpc'] = min(lin[k]['MPC'].get('term_ok_frac', np.nan) for k in lin)
    # linearization study
    R['linz'] = J('linearization.json')
    # S4 coupling
    dx, dphi, yres, thres = [], [], [], []
    for c in setup.ALL:
        a, b = run('nominal', 'S4', c), run('nominal', 'S1', c)
        dx.append(1000 * np.max(np.abs(a['xT'] - b['xT'])))
        dphi.append(np.degrees(np.max(np.abs(a['phi'] - b['phi']))))
        m = a['t'] >= a['t'][-1] - 5.0
        yres.append(np.max(np.abs(a['yc'][m])))
        thres.append(np.degrees(np.max(np.abs(a['theta'][m]))))
    R['S4_coupling'] = dict(dx_mm=float(max(dx)), dphi_deg=float(max(dphi)), y_res=float(np.mean(yres)),
                            th_res=float(np.mean(thres)))
    # influence of the trolley force on the lateral and torsional motion in S4 (spread over controllers)
    ref = run('nominal', 'S4', 'LQR')
    dy, dth = [], []
    for c in setup.ALL:
        a = run('nominal', 'S4', c)
        dy.append(1000 * np.max(np.abs(a['yc'] - ref['yc'])))
        dth.append(np.degrees(np.max(np.abs(a['theta'] - ref['theta']))))
    R['S4_coupling'].update(dy_mm=float(max(dy)), dth_deg=float(max(dth)))
    rt = J('rope_tension_check.json')['_summary']
    R['rope_min'], R['rope_min_case'], R['rope_max'] = rt['Tmin_kN'], rt['Tmin_case'], rt['Tmax_kN']
    # settling time in S1 versus the band half-width (the 2 % result rests on a small overshoot)
    from sim.metrics import settling_time
    R['band_sens'] = {}
    for c in ('PID (optimal)', 'MPC', 'LQR'):
        d = run('nominal', 'S1', c)
        R['band_sens'][c] = {f'{1000 * b:.0f}': float(settling_time(d['t'], d['xT'], b))
                             for b in (0.010, 0.011, 0.012, 0.015, 0.025)}
    R['po_os_mm'] = float(-1000 * run('nominal', 'S1', 'PID (optimal)')['xT'].min())
    R['phi_max_S1'] = max(R['nom']['S1'][c]['phi_peak'] for c in setup.ALL)
    R['phi_max_S2'] = max(R['nom']['S2'][c]['phi_peak'] for c in setup.ALL)
    # MPC-SC limits under model mismatch (Monte Carlo); admissible initial swing |phi(0)| <= 2 deg
    from scipy.stats import spearmanr
    keys = [k for k in R['mc'] if k.startswith('MC')]
    sc_rows = [R['mc'][k]['MPC-SC'] for k in keys]
    smp = {p: np.array([R['mc'][k]['_sample'][p] for k in keys]) for p in ('tau', 'l', 'phi0_deg')}
    adm = np.abs(smp['phi0_deg']) <= 2.0
    v = np.array([r['vT_peak'] for r in sc_rows])
    ph = np.array([r['phi_peak'] for r in sc_rows])
    tv = np.array([r.get('t_over_v', np.nan) for r in sc_rows])
    tp = np.array([r.get('t_over_phi', np.nan) for r in sc_rows])
    vo, po = adm & (v > 0.5001), adm & (ph > 2.0001)
    R['sc_mc'] = dict(n_adm=int(adm.sum()), n_v=int(vo.sum()), n_phi=int(po.sum()),
                      v_med=float(np.median(v[adm])), v_p95=float(np.percentile(v[adm], 95)),
                      tv_med=float(np.median(tv[vo])) if vo.any() else 0.0,
                      tv_max=float(np.max(tv[vo])) if vo.any() else 0.0,
                      phi_med_over=float(np.median(ph[po])) if po.any() else 0.0,
                      phi_p95=float(np.percentile(ph[adm], 95)),
                      tp_med=float(np.median(tp[po])) if po.any() else 0.0,
                      rho_v_tau=float(spearmanr(smp['tau'][adm], v[adm])[0]),
                      rho_phi_l=float(spearmanr(smp['l'][adm], ph[adm])[0]),
                      v_med_viol=float(np.median(v[vo])) if vo.any() else 0.0,
                      v_max_lowlag=float(np.max(v[adm & (smp['tau'] < 0.01)])),
                      n_lowlag=int(np.sum(adm & (smp['tau'] < 0.01))),
                      phi_bound_l4=float(np.degrees(np.arcsin(6.0 * np.sin(np.radians(2.0)) / 4.0))),
                      fallbacks=int(sum(R['mc'][k][c].get('fallbacks', 0) for k in keys for c in ('MPC-SC',))))
    R['term_mc'] = {c: dict(n_fail=int(sum(1 for k in keys if R['mc'][k][c].get('term_ok_frac', 1.0) < 1.0)),
                            min_frac=float(min(R['mc'][k][c].get('term_ok_frac', 1.0) for k in keys)))
                    for c in ('MPC', 'MPC-SC')}
    R['sc_slack_nom'] = float(max(R['nom'][k]['MPC-SC'].get('slack_max', 0.0) for k in ('S1', 'S2', 'S4', 'S5')))
    R['sc_slack_S3'] = float(R['nom']['S3']['MPC-SC'].get('slack_max', 0.0))
    R['sc_fallbacks_all'] = int(sum(R[g][k]['MPC-SC'].get('fallbacks', 0) for g in ('nom', 'rob') for k in R[g])
                                + R['sc_mc']['fallbacks'])
    # optional results of the additional checks
    for name, key in (('rope_angles.json', 'rope_angles'), ('limit_cycles.json', 'lc_long'),
                      ('trolley_integral.json', 'tri'), ('swing_integrator_bound.json', 'swb'),
                      ('linear_margins.json', 'margins'), ('lq_specs.json', 'lqs'),
                      ('equivalence_all.json', 'eqa'), ('friction_vs.json', 'friction_vs')):
        if os.path.exists(os.path.join(RES, name)):
            R[key] = J(name)
    if 'rope_angles' in R:
        ra = R['rope_angles']
        R['rope_S4_max'] = max(ra[f'S4/{c}']['rope_max_deg'] for c in setup.ALL)
        R['rope_S4_min'] = min(ra[f'S4/{c}']['rope_max_deg'] for c in setup.ALL)
        R['phi_S4_max'] = max(ra[f'S4/{c}']['phi_max_deg'] for c in setup.ALL)
        R['rope_S2_dev'] = max(ra[f'S2/{c}']['rope_minus_phi_max_deg'] for c in setup.ALL)
    qt = J('qp_timing.json')
    R['mpc_s2_timing'] = qt['MPC/S2']
    if 'total_mean_ms' in qt['MPC']:
        R['tot_mpc_mean'], R['tot_mpc_max'] = qt['MPC']['total_mean_ms'], qt['MPC']['total_max_ms']
        R['tot_sc_mean'], R['tot_sc_max'] = qt['MPC-SC']['total_mean_ms'], qt['MPC-SC']['total_max_ms']
    m = re.search(r'\((\d+) problems, (\d+) variables, (\d+) constraints\): converged at first attempt (\d+), '
                  r'restarts (\d+), converged after restart (\d+)', qv)
    if m:
        R['qp_real'] = dict(zip(('n', 'nvar', 'ncon', 'first', 'restarts', 'after'), map(int, m.groups())))
        k2 = re.search(r'MPC-SC KKT residuals \(relative\): max primal violation = ([0-9.e+-]+), '
                       r'max stationarity = ([0-9.e+-]+), max complementarity = ([0-9.e+-]+)', qv)
        R['qp_real'].update(viol=float(k2.group(1)), stat=float(k2.group(2)), comp=float(k2.group(3)))
        k3 = re.search(r'Random problems, IPM KKT residuals \(relative\): max primal violation = ([0-9.e+-]+), '
                       r'max stationarity = ([0-9.e+-]+)', qv)
        R['qp_rand_kkt'] = max(float(k3.group(1)), float(k3.group(2)))
    return R


if __name__ == '__main__':
    R = collect()
    for k, v in R.items():
        if k in ('nom', 'rob', 'lin', 'abl', 'eqv', 'mc', 'gains', 'specs', 'design'):
            continue
        print(k, v)
