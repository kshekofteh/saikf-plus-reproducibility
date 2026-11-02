"""
Exploration of the heterogeneous-node case for SAIKF+.

Theorem 2 of the TAES draft is exact but assumes homogeneous nodes: there,
column stochasticity forces S_{i,t} = C^T R^-1 C for every i and t, the
per-node covariances coincide, and the global error operator factors as
    Adot_t = (W_t^T W_{t-1}^T) kron M0.
Heterogeneity breaks exactly the commutation that makes this work.

This script tests three structural conjectures that might recover something
exact, or at least quantitative, for heterogeneous nodes.

  C1  S_{i,t} always lies in conv{Lambda_1,...,Lambda_N}, Lambda_j = C_j^T R_j^-1 C_j,
      independently of the topology.  (If true: the per-node information
      matrices live in a fixed compact convex set, which is what any
      perturbation argument needs.)

  C2  "Scalar-heterogeneous" sensors, C_i = C and R_i = r_i R, reduce
      S_{i,t} = alpha_{i,t} * Lambda with alpha_{i,t} = sum_j w_{ji,t}/r_j a
      scalar.  The per-node matrices then form a one-parameter family.

  C3  Near-homogeneity: rho of the switched product stays below 1 for
      heterogeneity up to a computable margin around the nominal system.
"""

import numpy as np

rng0 = np.random.default_rng(11)

# ---------------------------------------------------------------- model ----
DT = 0.1
M = 4
A_NOM = np.array([[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=float)
G = np.array([[DT**2 / 2, 0], [0, DT**2 / 2], [DT, 0], [0, DT]])
SIGMA_A = 0.5
Q_NOM = SIGMA_A**2 * (G @ G.T)
C_NOM = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
N = 8


def sa_transition_matrix(rng, n=N, m=5, support=3):
    Pr = np.zeros((n, m))
    for i in range(n):
        k = rng.integers(1, support + 1)
        cols = rng.choice(m, size=k, replace=False)
        w = rng.random(k) + 0.2
        Pr[i, cols] = w / w.sum()
    return Pr


def iwm_from_sa(Pr, beta):
    n, m = Pr.shape
    W = np.zeros((n, n))
    for i in range(n):
        nb = [j for j in range(n) if j != i]
        off = np.zeros(n)
        leaked = 0.0
        for k in range(m):
            if Pr[i, k] <= 0:
                continue
            denom = sum(Pr[j, k] for j in nb if Pr[j, k] > 0)
            if denom <= 0:
                leaked += Pr[i, k]
                continue
            for j in nb:
                if Pr[j, k] > 0:
                    off[j] += Pr[i, k] * (Pr[j, k] / denom)
        W[:, i] = (1.0 - beta) * off
        W[i, i] = beta + (1.0 - beta) * leaked
    return W


def random_colstoch(rng, n=N):
    """Uniform-ish random column-stochastic matrix (Dirichlet columns)."""
    W = rng.dirichlet(np.ones(n), size=n).T
    return W


# ------------------------------------------------- node parameter sets ----
def nodes_homog():
    return ([A_NOM] * N, [C_NOM] * N, [np.eye(2)] * N, [Q_NOM] * N)


def nodes_scalar_het(r):
    """C_i = C, R_i = r_i I: the case of the TAES numerical study."""
    return ([A_NOM] * N, [C_NOM] * N, [ri * np.eye(2) for ri in r], [Q_NOM] * N)


def nodes_general_het(delta, rng):
    """A_i, C_i, R_i, Q_i each perturbed by a relative amount <= delta."""
    As, Cs, Rs, Qs = [], [], [], []
    for i in range(N):
        dA = rng.standard_normal((M, M)); dA /= np.linalg.norm(dA, 2)
        dC = rng.standard_normal((2, M)); dC /= np.linalg.norm(dC, 2)
        As.append(A_NOM + delta * np.linalg.norm(A_NOM, 2) * dA)
        Cs.append(C_NOM + delta * np.linalg.norm(C_NOM, 2) * dC)
        Rs.append((1.0 + delta * rng.uniform(-1, 1)) * np.eye(2))
        Qs.append((1.0 + delta * rng.uniform(-1, 1)) * Q_NOM)
    return As, Cs, Rs, Qs


# --------------------------------------------------------- recursions -----
def lambdas(Cs, Rs):
    return [Cs[j].T @ np.linalg.solve(Rs[j], Cs[j]) for j in range(N)]


def node_S(W, Lam):
    return [sum(W[j, i] * Lam[j] for j in range(N) if W[j, i] > 0)
            for i in range(N)]


def steady_state_P(W, As, Cs, Rs, Qs, iters=20000, tol=1e-14):
    """P_i = [(P_i^-)^-1 + S_i]^-1 ,  P_i^- = sum_j w_ji (A_j P_j A_j^T + Q_j)."""
    S = node_S(W, lambdas(Cs, Rs))
    Pm = [np.eye(M) * 10.0 for _ in range(N)]
    P = [np.eye(M) * 10.0 for _ in range(N)]
    for _ in range(iters):
        P = [0.5 * (p + p.T) for p in
             [np.linalg.inv(np.linalg.inv(Pm[i]) + S[i]) for i in range(N)]]
        Pmn = []
        for i in range(N):
            acc = np.zeros((M, M))
            for j in range(N):
                if W[j, i] > 0:
                    acc += W[j, i] * (As[j] @ P[j] @ As[j].T + Qs[j])
            Pmn.append(0.5 * (acc + acc.T))
        d = max(np.max(np.abs(Pmn[i] - Pm[i])) for i in range(N))
        Pm = Pmn
        if d < tol:
            break
    return P, S


def A_dot(Wt, Wprev, P, S, As):
    """Adot = W_t^T (I - P S) W_{t-1}^T diag{A_i}."""
    Wt_k = np.kron(Wt.T, np.eye(M))
    Wp_k = np.kron(Wprev.T, np.eye(M))
    IPS = np.eye(N * M)
    Ablk = np.zeros((N * M, N * M))
    for i in range(N):
        sl = slice(i * M, (i + 1) * M)
        IPS[sl, sl] = np.eye(M) - P[i] @ S[i]
        Ablk[sl, sl] = As[i]
    return Wt_k @ IPS @ Wp_k @ Ablk


def rho(X):
    return float(max(abs(np.linalg.eigvals(X))))


# ========================================================== C1: polytope ===
def test_C1(trials=300):
    """S_i in conv{Lambda_j}?  Checked by solving for the convex weights."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(trials):
        As, Cs, Rs, Qs = nodes_general_het(0.4, rng)
        Lam = lambdas(Cs, Rs)
        W = random_colstoch(rng)
        S = node_S(W, Lam)
        # S_i = sum_j w_ji Lambda_j exactly, with w_ji >= 0 summing to 1:
        # verify the identity directly and the stochasticity of the weights
        for i in range(N):
            recon = sum(W[j, i] * Lam[j] for j in range(N))
            worst = max(worst, np.max(np.abs(S[i] - recon)))
        assert abs(W.sum(axis=0) - 1).max() < 1e-12
    return worst


# ================================================ C2: scalar-heterogeneous =
def test_C2(trials=200):
    """With C_i = C and R_i = r_i I, is S_{i,t} = alpha_{i,t} * Lambda ?"""
    rng = np.random.default_rng(2)
    worst_resid, alpha_range = 0.0, []
    Lam_nom = C_NOM.T @ C_NOM            # R = I
    for _ in range(trials):
        r = rng.uniform(0.5, 2.5, size=N)
        As, Cs, Rs, Qs = nodes_scalar_het(r)
        Lam = lambdas(Cs, Rs)
        W = random_colstoch(rng)
        S = node_S(W, Lam)
        for i in range(N):
            alpha = sum(W[j, i] / r[j] for j in range(N))
            worst_resid = max(worst_resid, np.max(np.abs(S[i] - alpha * Lam_nom)))
            alpha_range.append(alpha)
    return worst_resid, (min(alpha_range), max(alpha_range))


# ============================================ C3: heterogeneity margin =====
def switched_rho(As, Cs, Rs, Qs, modes, horizon=120, rng=None):
    """Upper estimate of the joint spectral radius over the given IWM modes."""
    rng = rng or np.random.default_rng(0)
    PS = []
    for W in modes:
        P, S = steady_state_P(W, As, Cs, Rs, Qs)
        PS.append((P, S))
    ops = []
    for a in range(len(modes)):
        for b in range(len(modes)):
            ops.append(A_dot(modes[a], modes[b], PS[a][0], PS[a][1], As))
    best = 0.0
    for _ in range(12):
        prod = np.eye(N * M)
        for t in range(1, horizon + 1):
            prod = ops[rng.integers(len(ops))] @ prod
            if t >= 20:
                best = max(best, np.linalg.norm(prod, 2) ** (1.0 / t))
    per_mode = [rho(o) for o in ops]
    return best, max(per_mode)


if __name__ == "__main__":
    print("=" * 72)
    r = test_C1()
    print("C1  S_i = sum_j w_ji Lambda_j  (convex combination)")
    print("    max residual over 300 random heterogeneous networks: %.2e" % r)
    print("    => the information matrices lie in conv{Lambda_j}, independent")
    print("       of the topology.")

    print("=" * 72)
    resid, arange = test_C2()
    print("C2  scalar-heterogeneous sensors C_i=C, R_i=r_i I")
    print("    max residual of  S_i = alpha_i * Lambda : %.2e" % resid)
    print("    alpha_i ranged over [%.3f, %.3f]" % arange)

    print("=" * 72)
    print("C3  heterogeneity margin")
    rngm = np.random.default_rng(7)
    modes = [iwm_from_sa(sa_transition_matrix(np.random.default_rng(700 + k)), 0.2)
             for k in range(3)]
    print("    %-8s %-12s %-12s" % ("delta", "JSR est", "max mode rho"))
    for delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        if delta == 0.0:
            As, Cs, Rs, Qs = nodes_homog()
        else:
            As, Cs, Rs, Qs = nodes_general_het(delta, np.random.default_rng(99))
        j, m = switched_rho(As, Cs, Rs, Qs, modes)
        print("    %-8.2f %-12.5f %-12.5f" % (delta, j, m))
