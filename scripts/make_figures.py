"""Figures 3-7 of the revised manuscript from results/runs and results/metrics_*.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sim import params as P, setup
from sim.linear import z_to_s

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG = os.path.join(os.path.dirname(__file__), '..', 'figures')
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 11, 'legend.fontsize': 8.3,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.grid': True, 'grid.alpha': 0.35,
    'grid.linewidth': 0.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

STY = {
    'PD': ('#1f77b4', '-', 1.3), 'PID (soft)': ('#2ca02c', '--', 1.3),
    'PID (optimal)': ('#d62728', '-', 1.6), 'PID (medium)': ('#ff7f0e', '-.', 1.3),
    'PID (hard)': ('#9467bd', ':', 1.7), 'LQR': ('#000000', '-', 1.8),
    'MPC': ('#0072b2', '-', 1.8), 'MPC-SC': ('#8c564b', '--', 1.8),
}
LABEL = {'LQR': 'LQR', 'MPC': 'MPC', 'MPC-SC': 'MPC-SC'}


def load(group, sc, name):
    f = os.path.join(RES, 'runs', group, sc, name.replace(' ', '_').replace('(', '').replace(')', '') + '.npz')
    return dict(np.load(f))


def metrics(group):
    return json.load(open(os.path.join(RES, f'metrics_{group}.json')))


def fig3():
    m = metrics('nominal')['S1']
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True)
    for nm in setup.CLASSICAL:
        d = load('nominal', 'S1', nm)
        c, ls, lw = STY[nm]
        ax[0].plot(d['t'], d['xT'], color=c, ls=ls, lw=lw, label=nm)
        ax[1].plot(d['t'], np.degrees(d['phi']), color=c, ls=ls, lw=lw, label=nm)
        ax[2].plot(d['t'], d['u'], color=c, ls=ls, lw=lw, label=nm)
    band = m['PD']['band']
    for s in (band, -band):
        ax[0].axhline(s, color='gray', lw=0.7, ls=':')
    ax[0].set_ylabel(r'Trolley position $x_\mathrm{T}$ (m)')
    ax[0].set_title('(a) Trolley position')
    ax[0].legend(ncol=3, loc='upper right', framealpha=0.92)
    ax[1].set_ylabel(r'Swing angle $\varphi$ (deg)')
    ax[1].set_title('(b) Swing angle')
    ax[2].set_ylabel(r'Control force $F$ (kN)')
    ax[2].set_title('(c) Control force')
    ax[2].set_ylim(-26, 11)
    ax[2].set_yticks([-10, -5, 0, 5, 10])
    for s in (10, -10):
        ax[2].axhline(s, color='red', lw=0.7, ls='--', alpha=0.5)
    ax[2].set_xlabel('Time (s)')
    ax[2].set_xlim(0, 30)
    # inset: first 1.5 s of the force, placed in the empty lower part of the panel
    ins = ax[2].inset_axes([0.55, 0.085, 0.42, 0.235])
    for nm in setup.CLASSICAL:
        d = load('nominal', 'S1', nm)
        c, ls, lw = STY[nm]
        ins.plot(d['t'], d['u'], color=c, ls=ls, lw=lw * 0.8)
    ins.set_xlim(0, 1.5)
    ins.set_ylim(-11, 11)
    ins.set_yticks([-10, 0, 10])
    ins.tick_params(labelsize=7, pad=1.5)
    ins.text(0.97, 0.88, '0-1.5 s', transform=ins.transAxes, ha='right', va='top', fontsize=7.5,
             bbox=dict(fc='white', ec='none', pad=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig3_classical_S1.png'))
    plt.close(fig)


def fig4():
    m = metrics('nominal')['S1']
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True)
    for nm in ('PID (optimal)', 'LQR', 'MPC'):
        d = load('nominal', 'S1', nm)
        c, ls, lw = STY[nm]
        if nm == 'LQR':
            ls, c = '--', '#555555'
        ts = m[nm]['Ts']
        ax[0].plot(d['t'], d['xT'], color=c, ls=ls, lw=lw, label=f'{nm}, $t_s$ = {ts:.2f} s')
        ax[1].plot(d['t'], np.degrees(d['phi']), color=c, ls=ls, lw=lw, label=nm)
        ax[2].plot(d['t'], d['u'], color=c, ls=ls, lw=lw, label=nm)
        if np.isfinite(ts):
            ax[0].axvline(ts, color=c, ls=':', lw=0.9)
    band = m['MPC']['band']
    for s in (band, -band):
        ax[0].axhline(s, color='gray', lw=0.7, ls=':')
    ax[0].set_ylabel(r'Trolley position $x_\mathrm{T}$ (m)')
    ax[0].set_title('(a) Trolley position')
    ax[0].legend(loc='upper right', framealpha=0.92)
    ax[1].set_ylabel(r'Swing angle $\varphi$ (deg)')
    ax[1].set_title('(b) Swing angle')
    ax[2].set_ylabel(r'Control force $F$ (kN)')
    ax[2].set_title('(c) Control force')
    ax[2].set_ylim(-11, 11)
    for s in (10, -10):
        ax[2].axhline(s, color='red', lw=0.7, ls='--', alpha=0.5)
    ax[2].set_xlabel('Time (s)')
    ax[2].set_xlim(0, 30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig4_MPC_PID_S1.png'))
    plt.close(fig)


def fig5():
    A, B, Ad, Bd = setup.design_model()
    Wc = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(4)])
    sv = np.linalg.svd(Wc, compute_uv=False)
    ol = np.linalg.eigvals(A)
    K, Pm, evz = setup.lqr_gain()
    evs = z_to_s(evz, P.TS)
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.5))
    ax[0].bar(range(1, 5), sv, color='#4a78a8', edgecolor='k', hatch='//')
    ax[0].set_yscale('log')
    ax[0].set_xticks(range(1, 5))
    ax[0].set_xlabel('Singular-value index')
    ax[0].set_ylabel('Magnitude (log scale)')
    ax[0].set_title('(a) Singular values of $\\mathcal{C}$')
    for i, v in enumerate(sv):
        ax[0].text(i + 1, v * 1.25, f'{v:.4g}', ha='center', fontsize=7.5)
    ax[0].set_ylim(sv.min() / 3, sv.max() * 4)
    ax[1].plot(ol.real, ol.imag, 'x', color='#c0392b', ms=11, mew=2.2)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-4.7, 4.7)
    ax[1].set_xlabel('Real (s$^{-1}$)')
    ax[1].set_ylabel('Imaginary (s$^{-1}$)')
    ax[1].set_title('(b) Open-loop poles')
    ax[1].axhline(0, color='gray', lw=0.6)
    ax[1].axvline(0, color='gray', lw=0.6)
    ax[1].text(0.08, 3.84, r'$\pm j\omega_c$', fontsize=8.5, va='center')
    ax[1].text(0.08, 0.35, 'double pole', fontsize=8)
    ax[2].plot(evs.real, evs.imag, 'x', color='#2e7d32', ms=11, mew=2.2)
    ax[2].set_xlabel('Real (s$^{-1}$)')
    ax[2].set_ylabel('Imaginary (s$^{-1}$)')
    ax[2].set_title('(c) LQR closed-loop poles')
    ax[2].axhline(0, color='gray', lw=0.6)
    ax[2].axvline(0, color='gray', lw=0.6)
    ax[2].set_ylim(-1.2, 1.2)
    ins = ax[2].inset_axes([0.12, 0.57, 0.55, 0.38])
    ins.plot(evs.real, evs.imag, 'x', color='#2e7d32', ms=7, mew=1.6)
    ins.set_xlim(-8, 0.5)
    ins.set_ylim(-0.6, 0.6)
    ins.axhline(0, color='gray', lw=0.5)
    ins.axvline(0, color='gray', lw=0.5)
    ins.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig5_controllability_poles.png'))
    plt.close(fig)
    return sv, ol, evs, K


def fig6():
    names = ('PID (optimal)', 'LQR', 'MPC', 'MPC-SC')
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 9.4), sharex=True)
    for nm in names:
        d = load('nominal', 'S2', nm)
        c, ls, lw = STY[nm]
        if nm == 'LQR':
            ls, c = '-.', '#555555'
        ax[0].plot(d['t'], d['xT'], color=c, ls=ls, lw=lw, label=nm)
        ax[1].plot(d['t'], d['vT'], color=c, ls=ls, lw=lw, label=nm)
        ax[2].plot(d['t'], np.degrees(d['phi']), color=c, ls=ls, lw=lw, label=nm)
        ax[3].plot(d['t'], d['u'], color=c, ls=ls, lw=lw, label=nm)
    for s in (setup.V_MAX, -setup.V_MAX):
        ax[1].axhline(s, color='#8c564b', lw=0.8, ls=':')
    for s in (2.0, -2.0):
        ax[2].axhline(s, color='#8c564b', lw=0.8, ls=':')
    ax[0].set_ylabel(r'$x_\mathrm{T}$ (m)')
    ax[0].set_title('(a) Trolley position')
    ax[0].legend(loc='upper right', framealpha=0.92)
    ax[1].set_ylabel(r'$\dot{x}_\mathrm{T}$ (m/s)')
    ax[1].set_title('(b) Trolley velocity (dotted: MPC-SC limit)')
    ax[2].set_ylabel(r'$\varphi$ (deg)')
    ax[2].set_title('(c) Swing angle (dotted: MPC-SC limit)')
    ax[3].set_ylabel(r'$F$ (kN)')
    ax[3].set_title('(d) Control force')
    ax[3].set_ylim(-11, 11)
    ax[3].set_xlabel('Time (s)')
    ax[3].set_xlim(0, 30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig6_large_move_S2.png'))
    plt.close(fig)


def fig7():
    mc = metrics('montecarlo')
    keys = [k for k in mc]
    names = setup.ALL
    lab = ['PD', 'PID\nsoft', 'PID\nopt.', 'PID\nmed.', 'PID\nhard', 'LQR', 'MPC', 'MPC\nSC']
    panels = [('Ts', 'Settling time $t_s$ (s)', None), ('phi_peak', r'Peak swing $|\varphi|_{\max}$ (deg)', None),
              ('work', 'Absolute mechanical work (kJ)', None), ('e_final_mm', r'$|x_\mathrm{T}(30\ \mathrm{s})|$ (mm)', 'log')]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6))
    for (key, yl, sc), ax, tl in zip(panels, axs.ravel(), ['(a)', '(b)', '(c)', '(d)']):
        ax.set_title(tl, loc='left', fontsize=10.5)
        data = []
        for nm in names:
            v = np.array([mc[k][nm][key] for k in keys])
            if key == 'Ts':
                v = np.where(np.isfinite(v), v, 31.0)
            if key == 'e_final_mm':
                v = np.maximum(v, 1e-3)
            data.append(v)
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', ms=3, alpha=0.5))
        for patch, nm in zip(bp['boxes'], names):
            patch.set_facecolor(STY[nm][0])
            patch.set_alpha(0.45)
        for med in bp['medians']:
            med.set_color('k')
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(lab, fontsize=8)
        ax.set_ylabel(yl)
        if sc:
            ax.set_yscale(sc)
        if key == 'Ts':
            ax.axhline(30, color='gray', lw=0.7, ls='--')
            ax.set_ylim(0, 34)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig7_monte_carlo.png'))
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1:] or ['3', '4', '5', '6', '7']
    for w in which:
        globals()['fig' + w]()
        print('Fig', w, 'done')
