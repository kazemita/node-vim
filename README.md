# NODE-VIM: neural-ODE Lagrange multipliers for the variational iteration method

Code, trained generators and results for the paper

> *Learning the Lagrange multiplier as a dynamical system: a neural-ODE variational iteration method for strongly nonlinear oscillators.*

The multiplier of the variational iteration method (VIM) is a kernel λ(t,s). Every kernel that satisfies the VIM
stationarity conditions is the transition kernel of a linear dynamic, λ(t,s) = −e₁ᵀΦ(t,s)e₂, so one VIM correction
is one forward solve of a linear IVP. A small network maps the current iterate to the coefficients of that dynamic
(with latent memory states); the correction is computed with RK4 cell maps composed by a parallel prefix scan.
NODE-VIM uses the learned multiplier far from the solution and Newton's (tangent) multiplier near it.

## Contents

| file | what it does |
|---|---|
| `osc.py` | problems (Duffing, van der Pol, pendulum), grid, one VIM step (`vim_step`: affine RK4 + prefix scan), generator network, all multipliers (`run`: classical, freqadj, newton, learned, hybridK, adapt = NODE-VIM), Anderson acceleration, transition matrices, multiplier maps, contraction-horizon curve |
| `train.py` | physics-only training (unrolled log-residual loss on random windows) |
| `evalcore.py` | helpers: loading generators, single-window and multistage (windowed) VIM |
| `evaluate_all.py` | main evaluation of one generator: single window, multistage tolerance / fixed-budget runs |
| `validate.py`, `add_v4_adapt.py` | numerical checks of the theory (error identity, secant multiplier, horizons, orders, random generators) |
| `lambda_maps.py` | multiplier kernels λ(t,s) and their distance to the exact (secant) kernel |
| `ablate_eval.py` | ablation metrics for all Duffing generators |
| `cost.py`, `count_learned.py`, `short_window.py` | cost per iteration, learned steps per window, short-window limitation study |
| `tables.py`, `make_ablation_tex.py`, `make_figs.py`, `fig_schematic.py` | tables and figures of the paper |
| `select_switch.py` | how the switching threshold r_sw was chosen |
| `run_ablations.sh` | trains all ablation variants and evaluates them |
| `models/` | trained generators (`gen_*.pt`) and training logs (`trainlog_*.json`) |
| `results/` | all JSON results used in the paper |

## Requirements

Python ≥ 3.10, PyTorch ≥ 2.0 (CPU is enough), NumPy, SciPy, Matplotlib (`pip install -r requirements.txt`).

## Reproducing the paper

Run from the repository root. The scripts read and write files in the working directory: to skip training,
copy `models/*` and `results/*` into the root first (`cp models/* results/* .`).

```bash
# 1. train the three generators (6-25 min each on one CPU core)
python train.py --prob duffing  --H 2 --N 200 --K 4 --m 2 --steps 1000 --plo 0.5 --phi 3.0
python train.py --prob vdp      --H 2 --N 200 --K 4 --m 2 --steps 1000 --plo 0.2 --phi 3.0
python train.py --prob pendulum --H 6 --N 600 --K 4 --m 2 --steps 1000 --plo 0.3 --phi 3.0

# 2. main evaluation (writes eval_*.json, sw_*.npz, ms_*.npz)
python evaluate_all.py gen_duffing_H2.00_K4_m2_classical_jri_ph1_s0.pt 20
python evaluate_all.py gen_vdp_H2.00_K4_m2_classical_jri_ph1_s0.pt 20
python evaluate_all.py gen_pendulum_H6.00_K4_m2_classical_jri_ph1_s0.pt 48

# 3. theory checks, multiplier maps, cost, limitations
python train.py --prob duffing --H 2 --N 200 --K 4 --m 0 --steps 1000   # m = 0 generator used by validate.py
python validate.py && python add_v4_adapt.py
python lambda_maps.py
python cost.py && python count_learned.py
python train.py --prob duffing --H 2 --N 200 --K 4 --m 2 --steps 1000 --feats sjri --Hmin 0.5
python short_window.py

# 4. ablations (Duffing)
./run_ablations.sh

# 5. tables and figures (written to paper2/)
python tables.py && python make_ablation_tex.py && python make_figs.py && python fig_schematic.py
```

The large intermediate arrays (`sw_*.npz`, `ms_*.npz`, about 100 MB) are not stored in the repository; step 2
regenerates them.

## Citation

Please cite the paper (citation to be added on publication). Companion work on Volterra integral equations:
<https://github.com/kazemita/vim>.

## License

MIT (see `LICENSE`).
