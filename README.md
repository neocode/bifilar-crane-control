# Predictive and classical control of a bifilar-suspended crane load

Simulation code and numerical results for a comparative study of PD/PID, LQR and model
predictive control (MPC) of a trolley that carries a long rigid beam on two ropes (bifilar
suspension). Every controller is evaluated with the same protocol on the same nonlinear
three-dimensional multibody plant. The protocol fixes the sampling period, the actuator limit,
the time horizon and the metric definitions. Every number, table and figure of the study is
regenerated from the scripts in this repository.

Authors: O. Podlesny, O. Sheremet.

> The manuscript itself, the correspondence with the journal and the reviewer reports are
> **not** part of this repository. Only code, numerical results and figures are included.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Requirements](#requirements)
3. [Models](#models)
4. [Controllers](#controllers)
5. [Scenarios](#scenarios)
6. [Metrics](#metrics)
7. [Reproducing the results](#reproducing-the-results)
8. [Output files](#output-files)
9. [Verification of the numerical tools](#verification-of-the-numerical-tools)
10. [Main results](#main-results)
11. [Notes and limitations](#notes-and-limitations)

---

## Repository layout

```
bifilar-crane-control/
├── sim/                     # simulation library (imported by all scripts)
│   ├── params.py            # nominal parameters, linear design model (A, B, C, C_p)
│   ├── plant.py             # nonlinear 3-D multibody plant (Numba), RK4 + ZOH input
│   ├── linear.py            # ZOH discretisation, discrete LQR / DARE, PID closed loop, z -> s
│   ├── controllers.py       # PD/PID with anti-windup, LQR, MPC, MPC-SC, saturated augmented LQR
│   ├── qp.py                # primal-dual interior-point QP solver (Mehrotra) + active-set polishing
│   ├── terminal.py          # maximal output-admissible set O_inf (Gilbert-Tan), membership test
│   ├── tuning.py            # specification-based PD/PID tuning (DE + Nelder-Mead)
│   ├── setup.py             # construction of all controllers (single source of truth for weights/gains)
│   ├── scenarios.py         # scenarios S1-S5, S2r, R1-R7 and Monte Carlo sampling
│   ├── simulate.py          # closed-loop simulation (nonlinear and linear plant)
│   └── metrics.py           # performance metrics (identical for all controllers)
├── scripts/                 # entry points: verification, tuning, experiments, figures
├── results/                 # all numerical results (JSON / TXT) and time series (runs/*.npz)
├── figures/                 # Figs. 1-7 (PNG, 300 dpi)
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.12. The results were produced with Python 3.12.3, NumPy 2.3.5, SciPy 1.17.0,
  Numba 0.63.1, Matplotlib 3.10.8 and Pillow 12.1.0 (see `requirements.txt`).
* Linux, a multi-core CPU. The experiment scripts start up to 20 worker processes; the
  reported run times refer to a 24-thread workstation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts set `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` before NumPy is
imported. Without this, every worker process would start its own BLAS thread pool and the
machine would be heavily oversubscribed.

All scripts are run from the repository root (`python scripts/<name>.py`). They locate `sim/`,
`results/` and `figures/` relative to their own path.

## Models

### Design model (used by LQR, MPC and the PID analysis)

Linear longitudinal model of the trolley and the beam centre of mass (CoM), small swing angle:

* State x = [x_T, ẋ_T, x_CoM, ẋ_CoM]ᵀ, where x_T is the trolley position and x_CoM is the
  horizontal position of the beam CoM relative to the trolley.
* Input u is the trolley drive force in kN.

```
A = [0 1        0        0;        B = [0;  1000/m1;  0; -1000/m1]
     0 0   m2 g/(m1 l)   0;
     0 0        0        1;        C = I_4,  C_p = [1 0 0 0; 0 0 1/l 0]  (outputs x_T and α = x_CoM / l)
     0 0  -(m1+m2) g/(m1 l) 0]
```

The model is discretised with a zero-order hold at T_s = 0.02 s (`sim/linear.py: c2d`).

### Evaluation plant (used for all reported results)

The evaluation plant (`sim/plant.py`) is a nonlinear multibody model of the trolley, two
inextensible ropes and a rigid beam that moves in three dimensions:

* Redundant coordinates p = [x_T, X, Y, Z, θ, ψ]: the trolley position, the CoM position of
  the beam, the yaw angle θ and the pitch angle ψ.
* Two holonomic rope constraints. The equations of motion are solved in KKT form, with
  Baumgarte stabilisation (α = β = 20 s⁻¹).
* Travel resistance of the trolley: a tanh-smoothed Coulomb term with μ = 8 N/kN and
  v_s = 2 mm/s, plus viscous resistance with c_v = 100 N·s/m.
* Relative damping of the beam motion: ζ = 0.5 % of critical, applied to the translational
  and torsional modes.
* Optional first-order lag of the drive force, τ (0 in the nominal case).
* Classical RK4 integration with step h = 0.5 ms. The controller output is held constant over
  each sampling period T_s = 0.02 s (ZOH), and the actuator saturates at |F| ≤ 10 kN.

### Nominal parameters (`sim/params.py`)

| Symbol | Value | Meaning |
|---|---|---|
| m1 | 500 kg | trolley mass |
| m2 | 4000 kg | beam mass |
| l | 6.0 m | rope length |
| 2a | 4.4 m | rope spacing on the beam (beam treated as a slender rod, J_C = m2 a²/3) |
| μ | 0.008 (8 N/kN) | specific travel resistance |
| c_v | 100 N·s/m | viscous drive / bearing resistance |
| ζ | 0.005 | relative damping of the suspended beam |
| F_max | 10 kN | actuator force limit |
| T_s | 0.02 s | sampling period, common to all controllers |
| T_end | 30 s | evaluation horizon |
| h | 0.5 ms | RK4 step |

Natural modes of the linearised plant: in-plane coupled mode 3.836 rad/s (1.638 s), lateral
swing 1.279 rad/s (4.914 s) and torsion √(3g/l) = 2.215 rad/s (2.837 s).

## Controllers

All controllers are built in `sim/setup.py`. This file is the only place where weights and
gains are defined.

| Name | Structure | Settings |
|---|---|---|
| PD | two-loop PD on x_T and α | tuned to the "optimal" specification without integral action |
| PID (soft / medium / optimal / hard) | two-loop PID on x_T and α with conditional-integration anti-windup and \|K_i ξ\| ≤ F_max | tuned to the specifications below |
| LQR | discrete state feedback | Q = diag(100, 10, 5000, 100), R = 0.01 |
| MPC | condensed MPC over the input sequence with a penalty on Δu; only the input constraint \|u\| ≤ F_max, solved exactly by BVLS | Q = diag(1000, 10, 5000, 100), R = 0.1, R_Δu = 0.5, N = 100 (2 s); terminal cost from the DARE of the augmented system; O_inf terminal set checked a posteriori |
| MPC-SC | as MPC, plus state limits \|α\| ≤ 2°, \|ẋ_T\| ≤ 0.5 m/s as soft constraints (ρ1 = 10⁷, ρ2 = 10⁵), solved by the interior-point QP of `sim/qp.py` | as MPC |
| SatLQR-MPCw | saturated augmented LQR with the MPC weights | used only in the MPC/LQR equivalence test |

### PD/PID tuning specifications (`sim/tuning.py`)

The gains are obtained by differential evolution followed by Nelder-Mead refinement. The
objective is the sum of two terms:

* the ITAE of the regulation test (x_T(0) = 0.5 m);
* the ITAE of a disturbance test (a step of −0.35 kN).

Violations of the following specifications are added to the objective as penalties:

| Variant | max. unsaturated force | max. overshoot | max. swing | min. damping ratio |
|---|---|---|---|---|
| PID (soft) | 2.5 kN | 5 % | 0.5° | 0.70 |
| PID (medium) | 5 kN | 10 % | 1.0° | 0.50 |
| PID (optimal) | 10 kN | 10 % | 1.5° | 0.40 |
| PID (hard) | saturation allowed | 20 % | 3.0° | 0.20 |
| PD | 10 kN | 10 % | 1.5° | 0.40 |

The resulting gains are stored in `results/gains.json` together with the design metrics of
each variant.

## Scenarios

Defined in `sim/scenarios.py`. In all scenarios the reference is x_T = 0. The initial offset
is given as x_T(0).

| Id | Description |
|---|---|
| S1 | x_T(0) = 0.5 m (reference case) |
| S2 | x_T(0) = 5 m (large repositioning, strong saturation) |
| S2r | S2 with a reference ramp of 0.5 m/s (a baseline for the classical controllers) |
| S3 | initial swing of 3°, x_T(0) = 0 |
| S4 | x_T(0) = 0.5 m, y_CoM(0) = 0.2 m, θ(0) = 5° (out-of-plane coupling) |
| S5 | 0.5 kN horizontal force on the beam from t = 2 s |
| R1 / R2 | m2 = 3000 kg / 5000 kg |
| R3 / R4 | l = 4 m / 8 m |
| R5 | no friction, no damping |
| R6 | travel resistance 16 N/kN |
| R7 | actuator lag τ = 0.1 s |
| MC | 200 random plants and initial conditions (seed 2026), uniform in: m2 ∈ [3000, 5000] kg, l ∈ [4, 8] m, μ ∈ [0, 0.016], ζ ∈ [0, 0.02], τ ∈ [0, 0.1] s, x_T(0) ∈ [0.25, 2] m, φ(0) ∈ [−3°, 3°] |

## Metrics

Defined in `sim/metrics.py` and computed in the same way for every controller.

| Key | Definition |
|---|---|
| `Ts` | settling time of x_T into the 5 % band, max(0.05·\|x_T(0)\|, 25 mm); `inf` means not settled within 30 s |
| `Ts2` | settling time into the 2 % band, max(0.02·\|x_T(0)\|, 10 mm) |
| `Ts_phi` | settling time of the swing angle into ±0.1° |
| `overshoot` | trolley overshoot, % of \|x_T(0)\| |
| `phi_peak`, `phi_res` | peak swing angle; maximum \|φ\| over the last 5 s (deg) |
| `Fpeak`, `work` | peak drive force (kN); work of the drive, ∫\|F\|·\|ẋ_T\| dt (kJ) |
| `IAE`, `ITAE` | ∫\|x_T\| dt, ∫ t\|x_T\| dt |
| `vT_peak` | peak trolley speed (m/s) |
| `Trope` | peak rope tension (kN) |
| `qp_*`, `term_ok_frac` | QP solve times; fraction of samples at which the terminal state lies in O_inf (MPC only) |

## Reproducing the results

Run the scripts from the repository root, in the order below. Each script overwrites its
output files in `results/` or `figures/`. The run times are wall-clock times measured on a
24-thread workstation.

```bash
source .venv/bin/activate

# 1. verification of the numerical tools
python scripts/check_plant.py          # eigenvalues, energy and constraint drift  -> results/plant_verification.txt
python scripts/check_qp.py             # IPM vs BVLS / SLSQP on random QPs          -> results/qp_verification.txt

# 2. PD/PID tuning (about 15 min, 20 processes)
python scripts/run_tuning.py           # -> results/gains.json

# 3. closed-loop experiments (about 5 min in total; mostly the Monte Carlo study)
python scripts/run_experiments.py      # all groups -> results/metrics_<group>.json, results/runs/<group>/
#   or a subset: python scripts/run_experiments.py nominal robust
#   groups: nominal, robust, linear, equivalence, ablation, ramp, montecarlo

# 4. additional checks
python scripts/check_integration.py    # RK4 (h = 0.5 / 0.25 ms) vs DOP853         -> results/integration_check.txt
python scripts/run_horizon_check.py    # MPC with N = 50 vs N = 100                -> results/horizon_check.json
python scripts/run_linearization.py    # linear design model vs nonlinear plant    -> results/linearization.json
python scripts/run_timing.py           # QP solve times (run on an idle machine)   -> results/qp_timing.json
python scripts/check_rope_tension.py   # minimum / maximum rope tension            -> results/rope_tension_check.json
python scripts/run_offset_sweep.py     # PID integrator saturation, MC correlations -> results/offset_sweep.json, results/mc_correlations.json
python scripts/run_pid_sensitivity.py  # sensitivity of PID (optimal) to l and φ(0) -> results/pid_sensitivity.json

# 5. figures
python scripts/make_fig2.py            # Fig. 2: block diagram of the evaluation protocol
python scripts/make_figures.py         # Figs. 3-7 from results/runs and results/metrics_*.json

# 6. summary of the numbers (gains, eigenvalues, statistics) printed to the console
python scripts/collect_results.py
```

Steps 4–6 read the outputs of steps 2–3. If only the figures are needed, step 5 can be run
directly on the results that are already in the repository.

Measured run times of `run_experiments.py`:

| Group | Runs | Time |
|---|---|---|
| nominal | 40 | 12 s |
| robust | 56 | 12 s |
| linear | 16 | 14 s |
| equivalence | 8 | 3 s |
| ablation | 18 | 1 s |
| ramp | 5 | 2 s |
| montecarlo | 1600 | 249 s |

The first call of each process includes the Numba JIT compilation of the plant.

## Output files

| File | Content | Produced by |
|---|---|---|
| `results/gains.json` | PD/PID gains, specifications, design metrics, optimiser settings | `run_tuning.py` |
| `results/metrics_nominal.json` | S1-S5, all controllers | `run_experiments.py nominal` |
| `results/metrics_robust.json` | R1-R7 | `run_experiments.py robust` |
| `results/metrics_linear.json` | S1 on the linear design model | `run_experiments.py linear` |
| `results/metrics_equivalence.json` | MPC vs saturated augmented LQR (S1, S2, S3, S5) | `run_experiments.py equivalence` |
| `results/metrics_ablation.json` | PID (optimal) with scaled integral gains (S1, S5) | `run_experiments.py ablation` |
| `results/metrics_ramp.json` | S2r (reference ramp) | `run_experiments.py ramp` |
| `results/metrics_montecarlo.json` | 200 samples × 8 controllers, sampled parameters in `_sample` | `run_experiments.py montecarlo` |
| `results/runs/<group>/<scenario>/<controller>.npz` | time series: `t`, `xT`, `vT`, `xc`, `vc`, `yc`, `theta`, `phi`, `u` (applied, kN), `u_un` (unsaturated), `Fact` (lagged drive force), `Trope` | `run_experiments.py` |
| `results/plant_verification.txt` | eigenvalues vs analytical frequencies, energy/constraint drift, static tensions | `check_plant.py` |
| `results/qp_verification.txt` | QP solver cross-check | `check_qp.py` |
| `results/integration_check.txt` | RK4 vs DOP853 | `check_integration.py` |
| `results/horizon_check.json` | effect of the MPC horizon | `run_horizon_check.py` |
| `results/linearization.json` | linear vs nonlinear responses | `run_linearization.py` |
| `results/qp_timing.json` | QP solve times per scenario | `run_timing.py` |
| `results/rope_tension_check.json` | rope tensions for all runs | `check_rope_tension.py` |
| `results/offset_sweep.json`, `results/mc_correlations.json` | PID integrator saturation vs initial offset; correlation of MC settling times with the sampled parameters | `run_offset_sweep.py` |
| `results/pid_sensitivity.json` | PID (optimal) vs rope length and initial swing | `run_pid_sensitivity.py` |
| `figures/Fig1_scheme.png` | scheme of the system | drawing (not generated by the code) |
| `figures/Fig2_block_diagram.png` | evaluation protocol | `make_fig2.py` |
| `figures/Fig3_classical_S1.png` | PD/PID/LQR responses, S1 | `make_figures.py` |
| `figures/Fig4_MPC_PID_S1.png` | MPC, MPC-SC and PID (optimal), S1 | `make_figures.py` |
| `figures/Fig5_controllability_poles.png` | controllability of the design model, open-loop and LQR closed-loop poles | `make_figures.py` |
| `figures/Fig6_large_move_S2.png` | large repositioning, S2 | `make_figures.py` |
| `figures/Fig7_monte_carlo.png` | Monte Carlo distributions of the metrics (box plots) | `make_figures.py` |

## Verification of the numerical tools

| Check | Result |
|---|---|
| Eigenvalues of the linearised multibody plant vs analytical frequencies | agree to 5 digits (1.27867, 2.21472, 3.83601 rad/s) |
| Free 3-D motion for 30 s, energy drift | 8·10⁻⁸ J (relative 3·10⁻¹³) |
| Rope constraint residual | 6·10⁻¹³ m |
| Static rope tensions | 19 620 N each (= m2 g/2) |
| RK4 (h = 0.5 ms) vs DOP853, S1/S2/S4 | max \|Δx_T\| ≤ 1.9·10⁻⁶ m, max \|Δφ\| ≤ 1.8·10⁻⁵ deg |
| Interior-point QP vs BVLS (40 box QPs, n = 30) | max \|Δx\| = 6·10⁻¹⁵ |
| Interior-point QP vs SLSQP (40 general QPs, 80 constraints) | max \|Δx\| = 1.5·10⁻⁶ |
| MPC horizon N = 50 vs N = 100 (S1-S3) | max \|Δu\| = 2·10⁻¹¹ kN; terminal state in O_inf at 100 % of samples |
| QP solve time (idle machine) | MPC: mean 0.99 ms, p99 1.04 ms; MPC-SC: mean 5.9 ms, p99 12.3 ms (T_s = 20 ms) |
| Minimum rope tension over all runs | 4.75 kN (S2, PID (hard)); the ropes never go slack |

## Main results

Reference case S1 (x_T(0) = 0.5 m), nonlinear plant, from `results/metrics_nominal.json`:

| Controller | Ts (5 %), s | Ts (2 %), s | Overshoot, % | φ peak, deg | F peak, kN | Work, kJ |
|---|---|---|---|---|---|---|
| PD | 4.14 | 7.80 | 0.0 | 1.43 | 10.00 | 0.24 |
| PID (soft) | 24.54 | 26.44 | 21.2 | 0.43 | 2.50 | 0.26 |
| PID (medium) | 13.84 | > 30 | 6.4 | 0.92 | 5.00 | 0.21 |
| PID (optimal) | 3.76 | 11.00 | 2.3 | 1.43 | 10.00 | 0.26 |
| PID (hard) | 3.04 | 3.24 | 0.9 | 2.96 | 10.00 | 1.57 |
| LQR | 6.66 | 10.68 | 2.2 | 0.50 | 10.00 | 0.21 |
| MPC | 6.62 | 7.78 | 6.6 | 1.39 | 10.00 | 0.48 |
| MPC-SC | 6.68 | 7.86 | 6.6 | 1.35 | 9.15 | 0.23 |

Main findings:

* **MPC and LQR.** With a Riccati terminal cost and only the input constraint active, the MPC
  coincides with the saturated augmented LQR that uses the same weights along the tested
  scenarios S1, S2, S3 and S5 (`metrics_equivalence.json`). The MPC therefore differs from linear state
  feedback only when state constraints are imposed (MPC-SC).
* **The settling band changes the ranking.** With the 5 % band the PID (optimal) settles
  faster than the MPC in S1 (3.8 s vs 6.6 s). With the 2 % band the order is reversed
  (11.0 s vs 7.8 s).
* **Swing limits in a large move.** In S2 only MPC-SC keeps |α| ≤ 2° and |ẋ_T| ≤ 0.5 m/s
  (φ peak 2.00°, work 4.5 kJ). The unconstrained controllers reach 3.9–21°. A reference ramp
  (S2r) gives the classical controllers a similar behaviour, but the limits are then not
  enforced.
* **Robustness of the PID.** The nominal advantage of the PID (optimal) is not robust. In
  the Monte Carlo study its median 5 % settling time is 11.2 s, against 6.7 s for LQR and
  6.9 s for MPC. The cause is that the two integrators saturate in opposition at the
  anti-windup bounds (`offset_sweep.json`). High-gain designs develop limit cycles when the
  drive lag is combined with low friction.
* **Structure of the two-loop PID.** The PID loop has a structurally uncontrollable pole at
  z = 1: the combination g·ξ_α + ẋ_T + ẋ_CoM is invariant. The closed-loop matrix from
  `sim/linear.py: pid_closed_loop` has an eigenvalue equal to 1 to machine precision.

## Notes and limitations

* The study is purely numerical. The evaluation plant is a higher-fidelity model than the
  design model, but it is not experimental validation.
* Monte Carlo time series are not stored. Only their metrics are saved, in
  `metrics_montecarlo.json`.
* QP timings depend on the machine and its load. `qp_*` fields in `metrics_*.json` were
  measured during parallel batch runs; the representative values are those in
  `results/qp_timing.json` (idle machine).
* Run logs (`*_log.txt`), `__pycache__` and Numba caches are not tracked (see `.gitignore`).
* No licence has been chosen yet. Until one is added, all rights are reserved by the authors.
