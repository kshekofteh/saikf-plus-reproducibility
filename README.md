# Reproducibility package

Code and generated data for the article

> M. Baradaran Khalkhali, S.-K. Shekofteh, A. Vahedian and H. Sadoghi Yazdi,
> "Mean-Square Stability of a Situation-Aware Interactive Kalman Filter over
> Time-Varying Interaction Networks, with Application to Multi-Vehicle
> Tracking," submitted to *IEEE Transactions on Aerospace and Electronic
> Systems*.

Every numerical value and every data-driven figure in Section VII of the
article is produced by the single script in this package. The study is seeded,
so the results are reproducible exactly.

## Contents

| File | Content |
|---|---|
| `saikf_study.py` | The complete study: all five experiments |
| `data/e1_msd_curve.dat` | Fig. 4(a) — Monte-Carlo vs closed-form network MSD |
| `data/e1_msd_per_node.dat` | Fig. 3 — per-node steady-state MSD |
| `data/e2_switching.dat` | Fig. 5(a) — MSD under switching topology |
| `data/e2_product_norm.dat` | Fig. 5(b) — decay of the accumulated product |
| `data/e3_beta.dat` | Fig. 5(c) — self-weight sweep |
| `data/e4_sa_vs_dist.dat` | Fig. 4(b) — SA-driven vs distance-only IWM |
| `data/e5_propagation.dat` | Fig. 4(c) — covariance-propagation rules |
| `data/results.json` | Every scalar quoted in the article |

## Running it

```
python3 saikf_study.py
```

Requires Python 3.9+ and NumPy; nothing else. Runtime is a few minutes on a
laptop. The script rewrites `data/` and prints a summary of each experiment.

## What the experiments are

The plant is the interactive model of the IKF framework: each target's next
state is the weight-combination of its neighbours' propagated states. Nodes
are heterogeneous (position-only measurements, standard deviations spread over
0.8–1.5 m). The interaction weight matrix is produced by the article's own
situation-assessment rule from randomly drawn transition probabilities over
five clusters, not by a generic graph model.

- **E1** Monte-Carlo validation of the closed-form steady-state network MSD.
- **E2** Uniform stability under a topology that switches at every step with
  no dwell time, plus a direct test of the article's Theorem 2 over 200 random
  topologies.
- **E3** Effect of the self-weight β on convergence rate and steady-state MSD.
- **E4** The diverging-neighbour mechanism: SA-driven versus distance-only
  interaction weights, with paired statistics.
- **E5** The three covariance-propagation rules compared.

## Key values to check

A correct run reproduces, among others:

| Quantity | Value |
|---|---|
| ρ(Ȧ) for the fixed SA configuration | 0.9527 |
| Closed-form network MSD | 0.0469 |
| Monte-Carlo network MSD (500 runs) | 0.0482 |
| Agreement | 2.79 % |
| Theorem 2 check: violations of ρ(Ȧ) ≤ ρ(M₀) | 0 of 200 |
| ρ with the squared-weight propagation rule | 0.99999 |
| ρ with the convex propagation rule | 0.9527 |

## Not included

The traffic-video evaluation of Section VIII is not part of this package. It
uses third-party material — publicly available surveillance clips and
sequences of the UA-DETRAC benchmark — which is not ours to redistribute. The
article states the limitations of that evaluation in its Section X.

## License

The code is released under the MIT License. The generated data files are
released under CC BY 4.0.

## Heterogeneous-sensor analysis (added for the TSP submission)

These scripts verify the lemmas and the heterogeneous-sensor theorem, and
generate the data for the heterogeneity figure.

| File | Purpose |
|---|---|
| `explore_heterogeneous.py` | Checks the structural conjectures: the information matrices lie in a fixed convex set, and for common dynamics with scalar-heterogeneous sensors the information reduces to a scalar multiple of a single matrix |
| `common_lyapunov.py` | Samples the reachable set of per-agent blocks and searches for a common quadratic Lyapunov function by derivative-free optimisation |
| `verify_theorem.py` | Verifies that the weighted block-maximum norm of the combination matrix is exactly one, and that the product envelope holds |
| `make_sweep_data.py` | Sweeps the sensor spread and writes `data/h1_spread.dat` |

Run order: `common_lyapunov.py`, then `verify_theorem.py`, then
`make_sweep_data.py`. Requires Python 3.9+ and NumPy.

Key values a correct run reproduces:

| Quantity | Value |
|---|---|
| Scalar reduction residual | 4e-16 |
| Weighted block-maximum norm of the combination matrix | 1.000000 |
| Common Lyapunov certificate | 0.9733, against max per-agent spectral radius 0.9687 |
| Product envelope ratio | 1.02 |
| Certificate satisfiable to a sensor spread of | 4 |
| Joint spectral radius below one to a spread of at least | 25 |

