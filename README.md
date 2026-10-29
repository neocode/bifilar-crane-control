# Predictive and classical control of a bifilar-suspended crane load

Simulation code and numerical results for a comparative study of PD/PID, LQR and model
predictive control (MPC) of a trolley that carries a long rigid beam on two ropes (bifilar
suspension). Every controller is evaluated with the same protocol on the same nonlinear
three-dimensional multibody plant. The protocol fixes the sampling period, the actuator limit,
the time horizon and the metric definitions. Every number, table and figure of the study is
regenerated from the scripts in this repository.

Authors: S. Podlesny, O. Sheremet.

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
7. [Numerical experiments](#numerical-experiments)
8. [Reproducing the results](#reproducing-the-results)
9. [Output files](#output-files)
10. [Verification of the numerical tools](#verification-of-the-numerical-tools)
11. [Main results](#main-results)
12. [Notes and limitations](#notes-and-limitations)

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

Stability (spectral radius below 1) and the damping specification are checked on the
eigenvalues of the discrete closed loop. The swing-angle integrator adds an uncontrollable
eigenvalue z = 1 for any gains (the discrete counterpart of the invariant g·ξ_α + ẋ_T + ẋ_CoM).
Its computed value equals 1 only to rounding accuracy, so it is removed analytically before the
check: the closed-loop matrix is restricted to the invariant subspace orthogonal to its
gain-independent left eigenvector (`sim/linear.py: pid_structural_mode, remove_structural_mode`).

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
| `phi_osc` | swing oscillation amplitude over the last 5 s (half the peak-to-peak value about the mean) |
| `t_over_phi`, `t_over_v`, `phi_excess`, `v_excess` | time spent above the MPC-SC limits (\|φ\| > 2°, \|ẋ_T\| > 0.5 m/s) and largest excess, reported for every controller |
| `qp_*`, `term_ok_frac` | QP solve times; fraction of samples at which the terminal state lies in O_inf (MPC only) |

## Numerical experiments

The study consists of a tuning stage (E1), twenty-one further experiments (E2–E22) and three
verification experiments for the numerical tools (V1–V3). All of them are run by the scripts in
`scripts/`; no other code is needed.

| Id | Experiment | Controllers | Runs | Script | Output |
|---|---|---|---|---|---|
| E1 | specification-based PD/PID tuning | PD, 4 × PID | 5 × 4 seeds + 20 restarts | `run_tuning.py` | `gains.json` |
| E2 | nominal comparison, S1-S5 | all 8 | 40 | `run_experiments.py nominal` | `metrics_nominal.json`, Figs. 3, 4, 6 |
| E3 | one-at-a-time parameter variations, R1-R7 | all 8 | 56 | `run_experiments.py robust` | `metrics_robust.json` |
| E4 | Monte Carlo robustness study | all 8 | 1600 | `run_experiments.py montecarlo` | `metrics_montecarlo.json`, Fig. 7 |
| E5 | same tests on the linear design model | all 8 | 16 | `run_experiments.py linear` | `metrics_linear.json` |
| E6 | MPC vs saturated augmented LQR | MPC, SatLQR-MPCw | 8 | `run_experiments.py equivalence` | `metrics_equivalence.json` |
| E7 | ablation of the integral action | PID (optimal) | 18 | `run_experiments.py ablation` | `metrics_ablation.json` |
| E8 | reference-ramp baseline, S2r | PD, PID (optimal), PID (hard), LQR, MPC | 5 | `run_experiments.py ramp` | `metrics_ramp.json` |
| E9 | validity range of the linear design model | PD, PID (optimal), LQR | 27 × 2 | `run_linearization.py` | `linearization.json` |
| E10 | MPC prediction horizon, N = 50 vs 100 | MPC | 6 | `run_horizon_check.py` | `horizon_check.json` |
| E11 | QP time, complete control time and terminal-set membership | MPC, MPC-SC | 10 | `run_timing.py` | `qp_timing.json` |
| E12 | rope tension (taut-rope assumption) | all 8 | 96 | `check_rope_tension.py` | `rope_tension_check.json` |
| E13 | initial-offset sweep; Monte Carlo rank correlations | PD, PID (optimal), LQR, MPC | 16 | `run_offset_sweep.py` | `offset_sweep.json`, `mc_correlations.json` |
| E14 | local sensitivity of the reference case | PD, PID (optimal), LQR, MPC | 36 | `run_pid_sensitivity.py` | `pid_sensitivity.json` |
| E15 | 300 s continuation of runs with possible sustained oscillations (force or swing) | flagged runs | ≈ 125 | `run_limit_cycles.py` | `limit_cycles.json` |
| E16 | PID with integral action in the trolley channel only (tuned as in E1) | PID (trolley I) | 6 tunings + 212 | `run_trolley_integral.py` | `trolley_integral.json` |
| E17 | role of the anti-windup bound of the swing integrator | PID (optimal) | 3 tests + 1 | `check_swing_integrator_bound.py` | `swing_integrator_bound.json` |
| E18 | inclination of the individual ropes vs the swing angle φ (S2, S4) | all 8 | 16 | `check_rope_angles.py` | `rope_angles.json` |
| E19 | linear stability margins: true rope length 4–8 m, drive lag 0–0.2 s | all except MPC-SC | linear | `check_linear_margins.py` | `linear_margins.json` |
| E20 | LQR and MPC evaluated with the PID tuning index and specifications | LQR, MPC | 2 × 2 | `check_lq_specs.py` | `lq_specs.json` |
| E21 | MPC vs saturated augmented LQR in all nominal, robustness and Monte Carlo cases | MPC, SatLQR-MPCw | 2 × 212 | `run_equivalence_all.py` | `equivalence_all.json` |
| E22 | sensitivity of S1 to the friction smoothing velocity v_s | PD, PID (optimal), LQR, MPC | 12 | `check_friction_regularization.py` | `friction_vs.json` |
| E23 | sensitivity of S1 to the saturation threshold of the conditional integration | PID (optimal) | 3 | `check_saturation_threshold.py` | `saturation_threshold.json` |
| V1 | plant verification | — | — | `check_plant.py` | `plant_verification.txt` |
| V2 | QP solver verification | — | 80 random + 4500 MPC-SC QPs | `check_qp.py` | `qp_verification.txt` |
| V3 | integration accuracy | PID (optimal), LQR, MPC, PID (hard) | 4 × 3 | `check_integration.py` | `integration_check.txt` |

### Common evaluation protocol

Every closed-loop run (E2–E18, E21–E23, V3) follows the same procedure (`sim/simulate.py: run_nonlinear`):

1. **Initial state.** The plant starts at rest in a consistent configuration. The trolley is at
   x_T(0). The beam CoM is placed according to the scenario: initial swing φ(0), lateral offset
   y_CoM(0) and yaw θ(0). The dependent coordinates are computed so that both rope constraints
   hold exactly (`plant.initial_state`).
2. **Task.** The trolley is brought from x_T(0) to x_T = 0 and the swing is damped (a
   point-to-point move). In S2r the controllers instead track the ramp
   x_ref(t) = sign(x_T(0))·max(|x_T(0)| − 0.5 t, 0).
3. **Measurement and control.** Every T_s = 0.02 s the controller receives
   [x_T, ẋ_T, x_CoM − x_T, ẋ_CoM − ẋ_T] (`plant.measured`). This is the full longitudinal state,
   without noise or delay. The controller returns the drive force, which is limited to ±10 kN
   and held constant until the next sample (ZOH).
4. **Plant.** Between samples the nonlinear plant is integrated by RK4 with h = 0.5 ms. The
   following act inside the plant: travel resistance, damping, the horizontal force on the beam
   (S5) and, where specified, the drive lag τ (R7, Monte Carlo).
5. **Evaluation.** Each run lasts 30 s. The metrics are computed from the logged signals by
   `sim/metrics.py`, identically for all controllers. Time series are stored in
   `results/runs/<group>/<scenario>/<controller>.npz` (except in E4).
6. **No retuning.** All controllers are designed once, for the nominal parameters. In R1-R7 and
   in the Monte Carlo study the same gains and the same MPC prediction model are applied to a
   plant with different parameters.

### E1. Specification-based PD/PID tuning (`run_tuning.py`, `sim/tuning.py`)

*Purpose.* To obtain PD and four PID variants, each tuned to its own explicitly stated
specifications (Section [PD/PID tuning specifications](#pdpid-tuning-specifications-simtuningpy)).
The comparison is then not tied to one arbitrary gain set.

*Closed loop used for tuning.* The design model is discretised exactly (ZOH, T_s = 0.02 s, 30 s
horizon). The loop includes the same saturation and conditional-integration anti-windup as the
evaluation runs. Each candidate gain vector (K_pT, K_iT, K_dT, K_pα, K_iα, K_dα) is simulated in
two tests:

* regulation from x_T(0) = 0.5 m, with all other states zero;
* the system at rest under a constant force of −0.35 kN on the trolley. This force is about the
  nominal Coulomb travel resistance: 0.008 · 4500 kg · 9.81 m/s² ≈ 0.35 kN.

*Objective.* J = ITAE(regulation) + ITAE(disturbance) + 100·v. Here v is the sum of normalised
violations of four specifications:

* peak unsaturated force (normalised by 0.5 kN);
* overshoot (by 1 %);
* peak swing (by 0.05°);
* minimum damping ratio of the closed-loop poles (by 0.05).

In addition, a candidate whose regulation response does not settle within 30 s gets a penalty of
one unit in v. Candidates whose remaining closed-loop eigenvalues (structural eigenvalue
removed, see above) have a spectral radius ≥ 1, or whose simulation diverges, are rejected.

*Search.*

* Bounds: K_pT ∈ [0, 200] kN/m, K_iT ∈ [0, 200] kN/(m·s), K_dT ∈ [0, 400] kN·s/m,
  K_pα ∈ [−3000, 3000] kN/rad, K_iα ∈ [−1000, 1000] kN/(rad·s), K_dα ∈ [−3000, 3000] kN·s/rad.
  PD has only the four proportional and derivative gains.
* Optimiser: differential evolution (SciPy `popsize=50` with Sobol initialisation, i.e. 512
  candidates for six gains and 256 for four or five; mutation 0.5–1.0, recombination 0.7; at most
  1000 generations), followed by restarted Nelder-Mead refinement. Each variant is run with 4
  random seeds.
* Second pass: each PID variant is restarted from the best solution of every other variant and of
  PD. The stochastic search therefore cannot return a worse optimum than a nested special case.
* The lowest-cost solution is kept. All costs found are stored in `gains.json` (`all_costs`),
  together with the design metrics of the selected gains. Runs that reached the best J to
  within 0.01 %: PD 4 of 4, soft 5 of 9, medium 9 of 9, optimal 9 of 9, hard 9 of 9.
* Numerical safeguards: the conditional integration treats the command as saturated only above
  F_max + 10⁻⁹ kN (the force budget of PD and PID (optimal) equals F_max, so their first command
  lies on the threshold), and integral gains below 10⁻⁹ are treated as zero, so that no
  admissibility decision depends on rounding.

### E2. Nominal comparison (`run_experiments.py nominal`)

All eight controllers are run in scenarios S1-S5 on the nominal plant (40 runs). This is the
main comparison. It gives settling times in the 5 % and 2 % bands, overshoot, peak and residual
swing, peak force, work, peak rope tension and the MPC solver statistics. Figures 3 and 4 show
S1 and Fig. 6 shows S2. S4 is compared with S1 to quantify the out-of-plane coupling: the extra
trolley and swing deviation caused by y_CoM(0) = 0.2 m and θ(0) = 5°. S5 tests the rejection of
a constant horizontal force on the beam.

### E3. Parameter variations (`run_experiments.py robust`)

All eight controllers are run with x_T(0) = 0.5 m on seven perturbed plants, R1-R7 (56 runs).
Each plant changes one parameter:

* beam mass ±25 %;
* rope length ±33 %;
* no dissipation at all;
* double travel resistance;
* a 0.1 s drive lag.

The controllers keep their nominal design.

### E4. Monte Carlo study (`run_experiments.py montecarlo`)

There are 200 random cases (seed 2026). In each case seven quantities are drawn simultaneously
and uniformly from the ranges in [Scenarios](#scenarios): m2, l, μ, ζ, τ, x_T(0) and φ(0). All
eight controllers are run on each case, 1600 runs in total. Only the metrics are stored, and the
sampled values are saved under `_sample`. A run that does not settle within 30 s has
`Ts = inf`. The distributions are shown in Fig. 7. The medians and percentiles are taken over
all 200 cases.

### E5. Linear design model as the plant (`run_experiments.py linear`)

S1 and S2 are repeated with the linear design model as the plant (16 runs). The model is
discretised exactly with 40 sub-steps per sample for the work integral. It has no friction, no
damping and no disturbance. Comparison with E2 shows how much of each controller's behaviour is
already explained by the linear model. In particular, it shows how much is caused by
dissipation and by the out-of-plane dynamics, which only the evaluation plant contains.

### E6. MPC versus saturated augmented LQR (`run_experiments.py equivalence`)

The MPC is compared with the saturated linear feedback u = sat(−K_a ξ) in S1, S2, S3 and S5
(8 runs). The feedback K_a is the terminal law of the MPC: the DARE solution for the state
augmented with u_{k−1}, which includes the R_Δu cross term. If the two trajectories coincide, the
input-constrained MPC adds nothing to linear state feedback with the same weights.

### E7. Ablation of the integral action (`run_experiments.py ablation`)

PID (optimal) is run with its two integral gains scaled by factors (c_T, c_α), giving
K_iT → c_T·K_iT and K_iα → c_α·K_iα (18 runs):

* (c_T, c_α) ∈ {(0, 0), (0, 1), (1, 0), (0.5, 0.5), (1, 1), (1.5, 1.5), (2, 2)} in S1 and S5;
* (c_T, c_α) ∈ {(0, 0), (0, 1), (1, 0), (1, 1)} in R6 (double travel resistance).

The experiment separates the role of each integrator in settling, in the residual position
error under friction and in the rejection of a constant force.

### E8. Reference-ramp baseline (`run_experiments.py ramp`)

S2 is repeated with a reference ramp of 0.5 m/s (S2r), equal to the trolley-speed limit of
MPC-SC (5 runs). It is run for PD, PID (optimal), PID (hard), LQR and MPC. This shows how much of
the MPC-SC behaviour in the large move can be obtained by simple reference shaping. The swing
and speed limits are then not enforced explicitly.

### E9. Validity range of the linear design model (`run_linearization.py`)

PD, PID (optimal) and LQR are each simulated twice: on the linear design model and on the
nonlinear plant. The nonlinear plant here has friction and damping removed, so that the
difference comes only from the geometric nonlinearity. The amplitude is increased through
x_T(0) ∈ {0.5, 1, 2, 3, 5} m and through φ(0) ∈ {2, 5, 10, 15}° (27 pairs of runs). The
experiment records three quantities: the peak swing, and the maximum deviations in x_T and φ
relative to their peaks. It also records the maximum difference in the control force.

### E10. MPC prediction horizon (`run_horizon_check.py`)

The MPC is run with N = 50 and N = 100 in S1-S3 on the nonlinear plant. The experiment records
the maximum difference between the applied forces and the fraction of samples at which the
predicted terminal state lies in the terminal set O_inf. The terminal set is not imposed as a
constraint; it is checked a posteriori. If the terminal state lies in O_inf, the Riccati terminal
cost equals the exact infinite-horizon cost, so the 2 s horizon is long enough.

### E11. QP solve time and terminal set (`run_timing.py`)

MPC and MPC-SC are run in S1-S5 in a single process, on an otherwise idle machine. The script
records the wall-clock time of every QP solve (set-up of the constraint data and solution; mean,
99th percentile and maximum), of every complete controller call (right-hand side of the QP, QP,
terminal-set check), the number of steps that exceed the sampling period, and the number of
solver iterations. It also checks a posteriori, at every sample, whether the predicted
terminal state lies in O_inf, and records the time from which this holds. The `qp_*` values in `metrics_*.json` come
from parallel batch runs and are not representative.

### E12. Rope tension (`check_rope_tension.py`)

All runs of E2 and E3 are repeated: 12 scenarios × 8 controllers = 96 runs. Both rope tensions
are recorded at every RK4 node (0.5 ms). The minimum over all runs must stay positive for the
inextensible-rope model to be valid. The maximum gives the peak rope load.

### E13. Initial-offset sweep and Monte Carlo correlations (`run_offset_sweep.py`)

*Offset sweep.* PD, PID (optimal), LQR and MPC are run on the nominal plant with
x_T(0) ∈ {0.5, 1.0, 1.5, 2.0} m (16 runs). The script records the settling time, the final
position error and the peak swing. For the PID it also records the final integral contributions
K_iT·ξ_T and K_iα·ξ_α. These show whether the two integrators end at their anti-windup bounds
±F_max with opposite signs, which explains the loss of performance of PID (optimal) in E4.

*Correlations.* The script computes Spearman rank correlations between the Monte Carlo settling
time of each controller and each sampled quantity, and also |l − 6 m|. Runs that do not settle
are assigned 40 s.

### E14. Local sensitivity of the reference case (`run_pid_sensitivity.py`)

S1 is repeated for PD, PID (optimal), LQR and MPC on a 3 × 3 grid (36 runs): rope length
l ∈ {5.5, 6.0, 6.5} m and initial swing φ(0) ∈ {−1.5°, 0°, +1.5°}. The experiment records the
settling time and the overshoot. It tests whether the nominal advantage of PID (optimal) in S1
persists under small model errors, or whether it is a sharp optimum.

### E15. Long-horizon check of sustained oscillations (`run_limit_cycles.py`)

All nominal, robustness and Monte Carlo runs with an RMS force above 1 kN over the last 10 s, or
a swing oscillation amplitude (`phi_osc`) above 0.1°, are repeated over 300 s with the same
plant, controller and initial condition. The script records the RMS force and the swing
amplitude in the windows 20–30, 90–100, 190–200 and 290–300 s, the periods, the fraction of time
at the force limit and the trolley amplitude. Classes: *limit cycle* (RMS force over 290–300 s
above 1 kN and within 10 % of its 20–30 s value), *sustained swing oscillation* (swing amplitude
still above 0.1° over 290–300 s), *decaying*. A run is *converged* if it settles within 30 s and
shows no sustained oscillation (Table 9 of the study).

### E16. PID with integral action in the trolley channel only (`run_trolley_integral.py`)

A PID with K_iα = 0 is tuned with the procedure of E1 and the specifications of PID (optimal)
(4 random seeds and 2 starts from related designs) and run in S1-S5, R1-R7 and the Monte Carlo
set. It tests whether trolley-channel integral action combined with load-velocity feedback
through the derivative gains can replace the two-channel PID.

### E17. Role of the anti-windup bound of the swing integrator (`check_swing_integrator_bound.py`)

Without saturation, the swing integrator equals an absolute load-velocity feedback with gain
−K_iα/g. The regulation test of E1 is repeated for PID (optimal) with and without the bound
|K_iα·ξ_α| ≤ F_max, and for its linear equivalent with integral action in the trolley channel
only. The number of samples at which the bound is active is also counted in S1 on the nonlinear
plant.

### E18. Inclination of the individual ropes (`check_rope_angles.py`)

S2 and S4 are repeated for all controllers. The script records the inclination of each rope
(AE, BD) to the vertical and the longitudinal swing angle φ used in the metrics and in the MPC-SC
constraint (angle of the trolley-CoM line in the vertical longitudinal plane). In planar motion
(S2) the three angles coincide; in S4 the lateral and torsional motion tilts the ropes.

### E19. Linear stability margins (`check_linear_margins.py`)

The discrete closed loops of the linear design model are formed with the true rope length l in
4–8 m while the controller keeps l_nom = 6 m, and with a first-order drive lag τ in 0–0.2 s at the
nominal rope length (the controllers measure x only, not the lagged force). The structural
eigenvalue z = 1 of the PID loops is excluded; the MPC is represented by its terminal law. The
script reports the rope lengths and the smallest lag at which an eigenvalue leaves the unit circle.

### E20. LQR and MPC evaluated with the PID tuning index (`check_lq_specs.py`)

The hand-tuned LQR and MPC (terminal law with u_prev) are simulated in the two linear tests of E1
and evaluated with the same quantities (J, peak unsaturated command, overshoot, peak swing,
minimum damping), to make the comparison with the specification-tuned PD/PID explicit.

### E21. MPC versus saturated LQR in all cases (`run_equivalence_all.py`)

MPC and SatLQR-MPCw are run on the nonlinear plant in S1–S5, R1–R7 and all 200 Monte Carlo
samples. The script records the largest command difference, the first and last time with a
difference above 10⁻⁶ kN, the largest difference before and after t = 1 s, and the differences of
the settling time and of the work.

### E22. Friction regularization (`check_friction_regularization.py`)

S1 is repeated for PD, PID (optimal), LQR and MPC with v_s = 2, 0.5 and 0.2 mm/s. The tanh
regularization has no stiction, so final errors below the settling band depend on v_s, whereas
the 5 % settling times do not.

### E23. Saturation threshold of the conditional integration (`check_saturation_threshold.py`)

The force budget of PID (optimal) equals F_max, so its first command in S1 lies about 10⁻⁹ kN
below the threshold F_max + 10⁻⁹ kN at which the conditional integration treats the command as
saturated. S1 is repeated with the tuned gains and with K_pT set 10⁻¹⁰ kN/m below and above the
value at which the first command reaches the threshold. Above it, the 5 % settling time is
unchanged (3.76 s), whereas the 2 % settling time, the overshoot and the ITAE change from 11.00 s,
2.33 % and 3.04 m·s² to 11.12 s, 2.37 % and 3.10 m·s².

### V1–V3. Verification of the numerical tools

* **V1, plant (`check_plant.py`).**
  * The conservative plant (μ = c_v = ζ = 0) is linearised numerically at the equilibrium
    (central differences). Its eigenvalues are compared with the analytical frequencies of the
    coupled, lateral and torsional modes.
  * A 30 s free 3-D motion is simulated, starting from x_CoM = 0.8 m, y_CoM = 0.5 m and
    θ = 10°. It checks energy conservation and the drift of the rope constraints.
  * The static rope tensions are compared with m2·g/2.
* **V2, QP solver (`check_qp.py`).**
  * 40 random strictly convex QPs are generated (n = 30, seed 0). In the box-constrained form
    they are solved by the interior-point solver and by SciPy BVLS. With 20 additional random
    general inequalities (80 constraints in total) they are solved by the interior-point solver
    and by SLSQP (ftol 10⁻¹⁰, the tightest tolerance at which SLSQP reports success on all
    problems). Every solver must report success.
  * The interior-point solutions are certified by their KKT residuals with a strict,
    per-constraint activity test (gap below 10⁻⁹(1 + |h_i|)): primal feasibility, stationarity
    with non-negative multipliers on the active set (NNLS) and complementarity.
  * The same KKT check is applied to all QPs of MPC-SC solved in S1-S3 (102 variables, 602
    inequality constraints); a negative test confirms that plans perturbed by 0.1 % fail it.
  * The BVLS solver of the input-constrained MPC runs with an iteration limit of 20N and its
    convergence status is checked at every solve.
* **V3, integration (`check_integration.py`).** Four runs (S1 with PID (optimal), S2 with LQR,
  S4 with MPC, R7 with PID (hard), the last with drive lag) are repeated three times: with RK4 at h = 0.5 ms, with RK4 at h = 0.25 ms, and
  with DOP853 (rtol 10⁻¹¹, atol 10⁻¹²) over every sampling interval with the same ZOH input.

The results are summarised in
[Verification of the numerical tools](#verification-of-the-numerical-tools).

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
python scripts/run_trolley_integral.py # PID with trolley-channel integral only    -> results/trolley_integral.json
python scripts/check_swing_integrator_bound.py  # role of the swing-integrator bound -> results/swing_integrator_bound.json
python scripts/check_rope_angles.py    # rope inclinations vs swing angle (S2, S4) -> results/rope_angles.json
python scripts/run_limit_cycles.py     # 300 s continuation of flagged runs        -> results/limit_cycles.json
python scripts/check_linear_margins.py # stability margins in l and drive lag      -> results/linear_margins.json
python scripts/check_lq_specs.py       # LQR/MPC with the PID tuning index         -> results/lq_specs.json
python scripts/run_equivalence_all.py  # MPC vs saturated LQR, all cases           -> results/equivalence_all.json
python scripts/check_friction_regularization.py  # S1 vs friction smoothing        -> results/friction_vs.json
python scripts/check_saturation_threshold.py     # S1 across the saturation threshold -> results/saturation_threshold.json

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
| `results/metrics_linear.json` | S1 and S2 on the linear design model | `run_experiments.py linear` |
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
| `results/trolley_integral.json` | gains, design metrics and S1-S5, R1-R7, Monte Carlo metrics of the PID with trolley-channel integral only | `run_trolley_integral.py` |
| `results/swing_integrator_bound.json` | regulation test of PID (optimal) with and without the swing-integrator bound; samples at the bound in S1 | `check_swing_integrator_bound.py` |
| `results/rope_angles.json` | maximum rope inclination and swing angle, S2 and S4 | `check_rope_angles.py` |
| `results/limit_cycles.json` | 300 s continuation and classification of the flagged runs | `run_limit_cycles.py` |
| `results/linear_margins.json` | rope-length and drive-lag stability limits of the linear loops | `check_linear_margins.py` |
| `results/lq_specs.json` | LQR and MPC evaluated with the PID tuning index | `check_lq_specs.py` |
| `results/equivalence_all.json` | MPC vs saturated LQR in all cases | `run_equivalence_all.py` |
| `results/friction_vs.json` | S1 metrics for three friction smoothing velocities | `check_friction_regularization.py` |
| `results/saturation_threshold.json` | S1 metrics of PID (optimal) below and above the saturation threshold | `check_saturation_threshold.py` |
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
| Interior-point QP vs SLSQP (40 general QPs, 80 constraints; all solvers successful) | max \|Δx\| = 1.4·10⁻⁶ (SLSQP tolerance); interior-point KKT residuals < 3·10⁻¹⁴ (strict test) |
| All MPC-SC QPs of S1-S3 (4500 problems, 102 variables, 602 constraints) | all converged at the first attempt; strict KKT residuals: stationarity < 2·10⁻¹¹, feasibility and complementarity < 3·10⁻¹⁴, up to 102 active constraints; plans perturbed by 0.1 % fail the test; the input-constrained fallback was never needed in any run |
| MPC horizon N = 50 vs N = 100 (S1-S3) | max \|Δu\| = 2·10⁻¹¹ kN; terminal state in O_inf at 100 % of samples |
| QP time per step (idle machine) | MPC: mean 1.00 ms, p99 1.16 ms; MPC-SC: mean 5.9 ms, p99 11.8 ms (T_s = 20 ms); complete control computation 1.03 and 6.0 ms on average; with BVLS solved to convergence (up to 143 iterations), 11 MPC steps, all in the first 0.22 s of S2, exceed 20 ms |
| Minimum rope tension over all runs | 4.75 kN (S2, PID (hard)); the ropes never go slack |

## Main results

Reference case S1 (x_T(0) = 0.5 m), nonlinear plant, from `results/metrics_nominal.json`:

| Controller | Ts (5 %), s | Ts (2 %), s | Overshoot, % | φ peak, deg | F peak, kN | Work, kJ |
|---|---|---|---|---|---|---|
| PD | 4.14 | 7.80 | 0.0 | 1.43 | 10.00 | 0.24 |
| PID (soft) | 24.56 | 26.44 | 21.2 | 0.43 | 2.50 | 0.26 |
| PID (medium) | 13.84 | > 30 | 6.4 | 0.92 | 5.00 | 0.21 |
| PID (optimal) | 3.76 | 11.00 | 2.3 | 1.43 | 10.00 | 0.26 |
| PID (hard) | 3.04 | 3.24 | 0.9 | 2.96 | 10.00 | 1.57 |
| LQR | 6.66 | 10.68 | 2.2 | 0.50 | 10.00 | 0.21 |
| MPC | 6.62 | 7.78 | 6.6 | 1.39 | 10.00 | 0.48 |
| MPC-SC | 6.68 | 7.86 | 6.6 | 1.35 | 9.15 | 0.22 |

Main findings:

* **MPC and LQR.** With a Riccati terminal cost and only the input constraint active, the MPC
  coincides with the saturated augmented LQR that uses the same weights in all nominal and
  robustness cases (difference ≤ 2·10⁻¹⁰ kN) and in 190 of the 200 Monte Carlo samples; in the
  other 10 the commands differ by up to 0.46 kN, starting in the saturated initial phase, with
  identical 5 % settling times and work differences of at most 0.03 kJ (`equivalence_all.json`). The MPC therefore differs from linear
  state feedback essentially only when state constraints are imposed (MPC-SC). The LQR and MPC
  weights were chosen by hand; evaluated with the PID tuning index, LQR gives J = 12.23
  (regulation / disturbance terms 4.83 / 7.40) and the MPC terminal law 7.09 (2.01 / 5.08),
  against 1.56 (1.28 / 0.28) for PID (optimal) and 9.10 (1.26 / 7.83) for PD (`lq_specs.json`).
* **The settling band changes the ranking.** With the 5 % band the PID (optimal) settles
  faster than the MPC in S1 (3.8 s vs 6.6 s). With the 2 % band the order is reversed
  (11.0 s vs 7.8 s), but only because the PID overshoot of 11.6 mm (2.3 %) exceeds the 10 mm
  band by 1.6 mm; for bands of 12–25 mm the PID is faster. The 2 % result of PID (optimal) is
  also sensitive to the saturation threshold: raising K_pT by about 2·10⁻⁹ kN/m changes it to
  11.12 s (`saturation_threshold.json`). Final errors below the settling band depend on the
  friction smoothing velocity (no stiction in the tanh model; `friction_vs.json`).
* **Swing limits in a large move.** In S2 only MPC-SC keeps |α| ≤ 2° and |ẋ_T| ≤ 0.5 m/s
  (φ peak 2.00°, work 4.5 kJ). The unconstrained controllers reach 3.9–21°. A reference ramp
  (S2r) gives the classical controllers a similar behaviour, but the limits are then not
  enforced. Under model mismatch (Monte Carlo) MPC-SC does not enforce its limits: of the 137
  samples with an admissible initial swing, the speed limit was exceeded in 129 (mainly because
  of the unmodelled drive lag) and the swing limit in 69 (mainly because the constraint uses the
  nominal rope length).
* **Robustness of the PID.** The nominal advantage of the PID (optimal) is not robust. In
  the Monte Carlo study its median 5 % settling time is 11.2 s, against 6.7 s for LQR and
  6.9 s for MPC. The cause is that the two integrators saturate in opposition at the
  anti-windup bounds (`offset_sweep.json`). Its nominal response relies on the active
  anti-windup bound of the swing integrator: without it, the overshoot in the tuning test rises
  from 2.1 % to 39 % (`swing_integrator_bound.json`). A PID with integral action in the trolley
  channel only, tuned to the same specifications, reduces to PD (`trolley_integral.json`).
* **Limit cycles.** High-gain designs (PID (hard), LQR, MPC, MPC-SC in 14, 5, 3 and 2 of the
  200 Monte Carlo samples) developed limit cycles only in samples that combine a drive lag of at
  least 0.056 s with a travel resistance of at most 5 N/kN, and not in all of them: of the samples
  with a lag above the linear limit of the loop (below) and such low friction, 14 of 27 (PID
  (hard)), 5 of 14 (LQR), 3 of 7 (MPC) and 2 of 7 (MPC-SC). Continued to 300 s, they keep their RMS force (7.3–8.8 kN) to within 1 %
  (period 0.18–0.42 s); one further flagged run decayed (`limit_cycles.json`). Their linear loops
  lose stability for drive lags above 0.050 s (PID (hard)), 0.074 s (LQR) and 0.082 s (MPC
  terminal law), close to the smallest lags at which limit cycles occurred (`linear_margins.json`).
* **Long ropes.** Designed for l = 6 m, PID (soft), PID (medium) and PID (optimal) lose linear
  stability for l ≥ 6.77, 7.21 and 7.85 m. PID (soft) in 40 of the 60 and PID (medium) in 10 of
  the 34 Monte Carlo samples beyond these limits, and both in R4 (l = 8 m), end in sustained swing
  oscillations (0.35–1.55°) whose force RMS stays below 1 kN; PID (optimal) is stabilized by the
  dissipation (no sustained oscillation in R4 or in the 4 samples beyond its limit). In R4 the
  oscillation periods (about 5 and 7 s) differ from those of the unstable linear modes (16.4 and
  10.4 s), so the linear analysis predicts the onset but not the form of these oscillations.
* **Structure of the two-loop PID.** The PID loop has a structurally uncontrollable pole at
  z = 1: the combination g·ξ_α + ẋ_T + ẋ_CoM is invariant. Its exact discrete left eigenvector
  is computed in `sim/linear.py: pid_structural_mode`.

## Notes and limitations

* The study is purely numerical. The evaluation plant is a higher-fidelity model than the
  design model, but it is not experimental validation.
* Monte Carlo time series are not stored. Only their metrics are saved, in
  `metrics_montecarlo.json`.
* QP timings depend on the machine and its load. `qp_*` fields in `metrics_*.json` were
  measured during parallel batch runs; the representative values are those in
  `results/qp_timing.json` (idle machine).
* Run logs (`*_log.txt`), `__pycache__` and Numba caches are not tracked (see `.gitignore`).
* Third round of checks (2026-10-29): Fig. 7 labels corrected (the
  PID (medium) and PID (optimal) boxes were swapped); BVLS iteration limit raised and its status
  checked (five solves of S2 had stopped at the SciPy default limit, with unchanged applied
  commands); KKT certificate made strict (the first version used an activity tolerance that
  could not fail); saturation threshold with a 10⁻⁹ kN tolerance; sustained swing oscillations
  detected and verified over 300 s; overshoot sign for negative offsets fixed (no such cases are
  used). The tuning and all experiments were re-run; the gains changed by less than 10⁻⁷.
  Experiments E19–E23 were added; the conditions under which sustained oscillations occur are
  reported as necessary, not sufficient (counts of the samples beyond the linear limits).
* Change of the tuning check (2026-10-29): the first version tested the spectral radius of the
  PID closed loop including the structural eigenvalue z = 1. Because that eigenvalue equals 1
  only to rounding accuracy, about half of all candidates were rejected at random, depending on
  the floating-point environment. With the corrected test the tuning reproduced the earlier
  optima (costs within 3·10⁻⁵, gains within 2·10⁻⁴, relative); all results were regenerated.
* No licence has been chosen yet. Until one is added, all rights are reserved by the authors.
