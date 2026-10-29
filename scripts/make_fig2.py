"""Fig. 2: closed-loop architecture used in the revised study."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavuserif'})
D = os.path.join(os.path.dirname(__file__), '..', 'figures')

fig, ax = plt.subplots(figsize=(10.0, 6.8))
ax.set_xlim(0, 100)
ax.set_ylim(-4, 66)
ax.axis('off')
LS = 4.4    # line spacing


def box(x, y, w, h, title, lines, fc, fs=10.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0', fc=fc, ec='#222222', lw=1.4))
    n = len(lines)
    top = y + h / 2 + (n * LS + 5.0) / 2
    ax.text(x + w / 2, top, title, ha='center', va='top', fontsize=12, weight='bold')
    for i, s in enumerate(lines):
        ax.text(x + w / 2, top - 6.2 - LS * i, s, ha='center', va='top', fontsize=fs)


def arrow(x0, y0, x1, y1, label=None, lx=0, ly=1.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#222222', mutation_scale=18))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha='center', va='bottom', fontsize=12)


YC = 33.0
box(1.5, 47, 19, 16, 'Reference', [r'$x_{\mathrm{T,ref}}=0$', r'$\alpha_{\mathrm{ref}}=0$'], '#EAF2FB')
ax.add_patch(Circle((11, YC), 3.4, fc='#FFF6D8', ec='#222222', lw=1.4))
ax.text(11, YC, r'$\Sigma$', ha='center', va='center', fontsize=15)
ax.text(7.6, YC + 4.6, '+', fontsize=13, weight='bold')
ax.text(6.9, YC - 8.6, '\u2212', fontsize=15, weight='bold')
arrow(11, 47, 11, YC + 3.6)

box(22.5, 19, 30, 28, 'Controller', ['PD, PID (four tunings)', 'LQR (discrete time)',
                                      r'MPC: $N=100$, terminal cost',
                                      'MPC-SC: MPC with swing', 'and speed constraints'], '#EFE7F5')
arrow(14.4, YC, 22.5, YC, r'$e$')

box(56, 25, 11.5, 16, 'ZOH', [r'$T_s=0.02$ s', r'$|u|\leq 10$ kN'], '#F3F3F3', fs=9.5)
arrow(52.5, YC, 56, YC, r'$u$')

box(71, 19, 27.5, 28, 'Plant', ['trolley, two ropes, beam', 'nonlinear 3-D', 'multibody model',
                                 'travel resistance', 'rope damping'], '#FDF1CC')
arrow(67.5, YC, 71, YC, r'$F$')

box(71, 51, 27.5, 12, 'Regulated outputs', [r'$x_\mathrm{T}(t),\ \alpha(t)=x_\mathrm{CoM}(t)/l$'], '#EAF2FB')
arrow(84.75, 47, 84.75, 51)

box(71, 1, 27.5, 14, 'Measurement', [r'$\mathbf{y}=C\mathbf{x},\ C=I_4$'], '#F3F3F3')
arrow(84.75, 19, 84.75, 15)
ax.plot([71, 11], [8.0, 8.0], color='#222222', lw=1.6)
arrow(11, 8.0, 11, YC - 3.6)
ax.text(41, 9.0, r'$\mathbf{x}(kT_s)=[x_\mathrm{T},\ \dot{x}_\mathrm{T},\ x_\mathrm{CoM},\ \dot{x}_\mathrm{CoM}]^\mathsf{T}$',
        ha='center', va='bottom', fontsize=11.5)
ax.text(50, -3.6, 'Lateral sway and torsion are simulated in the plant but are not fed back (single trolley force).',
        ha='center', va='bottom', fontsize=10, style='italic')
fig.savefig(os.path.join(D, 'Fig2_block_diagram.png'), dpi=300, bbox_inches='tight', facecolor='white')
print('saved')
