"""
Verification of the claimed heterogeneous-sensor theorem.

Two things have to hold for the argument to work.

  V1  No condition-number penalty.  In the block-maximum norm built from the
      SAME V at every node,  || W^T kron I ||  =  1  for every column-stochastic
      W.  (This is what fails when the nodes need different coordinates.)

  V2  The product bound.  With the common V of common_lyapunov.py,
      || Adot_t ... Adot_1 ||_V  <=  lambda^t  for every switching sequence.

It also sweeps the spread of the sensor noises to locate where a common V
stops existing, which is the practical content of the theorem.
"""

import numpy as np
from common_lyapunov import (A, Q, C, LAM, M, N, R_RANGE, random_colstoch,
                             sa_matrix, iwm_from_sa, steady_state,
                             collect_reachable, weighted_norm, minimise_lambda)


def block_max_norm_of_W(W, V, probes=4000, seed=0):
    """Empirical induced norm of W^T kron I in the V-weighted block-max norm.

    ||x||_{V,inf} = max_i sqrt(x_i^T V x_i).
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(V)
    worst = 0.0
    Wt = W.T
    for _ in range(probes):
        X = rng.standard_normal((N, M))
        nx = max(np.linalg.norm(L.T @ X[i]) for i in range(N))
        X = X / nx
        Y = Wt @ X                      # block row i: sum_j w_ji x_j
        ny = max(np.linalg.norm(L.T @ Y[i]) for i in range(N))
        worst = max(worst, ny)
    return worst


def Adot(Wt, Wprev, P, alpha):
    Wt_k = np.kron(Wt.T, np.eye(M))
    Wp_k = np.kron(Wprev.T, np.eye(M))
    Mblk = np.zeros((N * M, N * M))
    Ablk = np.kron(np.eye(N), A)
    for i in range(N):
        sl = slice(i * M, (i + 1) * M)
        Mblk[sl, sl] = np.eye(M) - P[i] @ (alpha[i] * LAM)
    return Wt_k @ Mblk @ Wp_k @ Ablk


def product_bound(V, lam, r, n_modes=6, horizon=120, seed=3):
    """Longest-run check of || Adot_t ... Adot_1 ||_V <= c lambda^t."""
    rng = np.random.default_rng(seed)
    modes = [iwm_from_sa(sa_matrix(np.random.default_rng(900 + k))) for k in range(n_modes)]
    PS = [steady_state(W, r) for W in modes]
    L = np.linalg.cholesky(V)
    Lbig = np.kron(np.eye(N), L)
    Lbig_inv = np.linalg.inv(Lbig)
    worst_ratio = 0.0
    for _ in range(8):
        seq = rng.integers(0, n_modes, size=horizon + 1)
        prod = np.eye(N * M)
        for t in range(1, horizon + 1):
            a, b = seq[t], seq[t - 1]
            prod = Adot(modes[a], modes[b], PS[a][0], PS[a][1]) @ prod
            nrm = np.linalg.norm(Lbig.T @ prod @ Lbig_inv.T, 2)
            worst_ratio = max(worst_ratio, nrm / lam**t)
    return worst_ratio


def sweep_spread():
    """How much sensor heterogeneity can a common V absorb?"""
    out = []
    for width in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        lo = 1.0 / np.sqrt(1 + width) if width else 1.0
        hi = 1.0 * np.sqrt(1 + width) if width else 1.0
        import common_lyapunov as cl
        cl.R_RANGE = (lo**2, hi**2)
        MM_all, MM, arange, trange, rh = cl.collect_reachable(trials=40, seed=5, keep=60)
        V, lam = cl.minimise_lambda(MM, iters=2500, seed=1)
        lam_all = cl.weighted_norm(V, MM_all)
        out.append((hi / lo, arange, rh.max(), lam_all))
    return out


if __name__ == "__main__":
    V = np.load("common_V.npy")
    lam = float(np.load("common_lambda.npy")[0])
    print("=" * 74)
    print("V1  induced norm of W^T kron I in the V-weighted block-max norm")
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(40):
        W = random_colstoch(rng) if k % 2 else iwm_from_sa(sa_matrix(rng))
        worst = max(worst, block_max_norm_of_W(W, V, probes=1500, seed=k))
    print("    max over 40 column-stochastic matrices : %.6f" % worst)
    print("    (the claim is that this equals 1 exactly, with no cond(V) penalty)")

    print("=" * 74)
    print("V2  product bound with lambda = %.5f" % lam)
    rng2 = np.random.default_rng(12)
    r = rng2.uniform(*R_RANGE, size=N)
    ratio = product_bound(V, lam, r)
    print("    max_t ||Adot_t...Adot_1||_V / lambda^t over 8 random sequences")
    print("    of length 120 : %.4f   (bounded => the envelope holds)" % ratio)

    print("=" * 74)
    print("heterogeneity sweep: how wide can the sensor spread be?")
    print("    %-12s %-22s %-12s %-10s" % ("sigma_hi/lo", "alpha range", "max rho(M)", "lambda"))
    for ratio_s, arange, rhmax, lam_w in sweep_spread():
        print("    %-12.1f [%.4f, %.4f]      %-12.5f %-10.5f"
              % (ratio_s, arange[0], arange[1], rhmax, lam_w))
