"""
Heterogeneous sensors, common dynamics: a candidate strengthening of Theorem 2.

Setting.  A_i = A, Q_i = Q, C_i = C for every node; only the measurement noise
differs, R_i = r_i R.  This is the regime of the application (identical vehicle
kinematics, cameras of differing quality) and of the numerical study.

Structure.  S_{i,t} = alpha_{i,t} Lambda with alpha_{i,t} = sum_j w_{ji,t}/r_j
a scalar in [1/r_max, 1/r_min], and since A is common, I kron A commutes with
W^T kron I, so

    Adot_t = W_t^T diag{M_{i,t}} W_{t-1}^T ,    M_{i,t} = (I - P_{i,t}S_{i,t}) A

with every block in ONE coordinate system.

Condition tested.  Seek V > 0 and lambda < 1 with M^T V M <= lambda^2 V for
every reachable block M.  Writing V = L L^T this is exactly

    || L^T M L^{-T} ||_2  <=  lambda      for all M,

i.e. a common contraction in the V-weighted norm.  Because the weighted
block-maximum norm built from the SAME V at every node satisfies
|| W^T kron I || = 1 for every column-stochastic W, with no condition-number
penalty, such a V gives

    || Adot_t ... Adot_1 ||  <=  c lambda^t

uniformly over every admissible switching sequence.
"""

import numpy as np

DT = 0.1
M = 4
A = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
G = np.array([[DT**2 / 2, 0], [0, DT**2 / 2], [DT, 0], [0, DT]])
Q = 0.5**2 * (G @ G.T)
C = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
LAM = C.T @ C
N = 8
R_RANGE = (0.64, 2.25)            # r_i = sigma_i^2, sigma_i in [0.8, 1.5]


def random_colstoch(rng, n=N):
    return rng.dirichlet(np.ones(n), size=n).T


def sa_matrix(rng, n=N, m=5, support=3):
    Pr = np.zeros((n, m))
    for i in range(n):
        k = rng.integers(1, support + 1)
        cols = rng.choice(m, size=k, replace=False)
        w = rng.random(k) + 0.2
        Pr[i, cols] = w / w.sum()
    return Pr


def iwm_from_sa(Pr, beta=0.2):
    n, m = Pr.shape
    W = np.zeros((n, n))
    for i in range(n):
        nb = [j for j in range(n) if j != i]
        off = np.zeros(n); leaked = 0.0
        for k in range(m):
            if Pr[i, k] <= 0:
                continue
            den = sum(Pr[j, k] for j in nb if Pr[j, k] > 0)
            if den <= 0:
                leaked += Pr[i, k]; continue
            for j in nb:
                if Pr[j, k] > 0:
                    off[j] += Pr[i, k] * (Pr[j, k] / den)
        W[:, i] = (1 - beta) * off
        W[i, i] = beta + (1 - beta) * leaked
    return W


def steady_state(W, r, iters=3000, tol=1e-13):
    alpha = (W / r[:, None]).sum(axis=0)              # alpha_i = sum_j w_ji / r_j
    S = [a * LAM for a in alpha]
    Pm = [np.eye(M) * 10.0 for _ in range(N)]
    P = list(Pm)
    for _ in range(iters):
        P = [0.5 * (p + p.T) for p in
             [np.linalg.inv(np.linalg.inv(Pm[i]) + S[i]) for i in range(N)]]
        APA = [A @ P[j] @ A.T + Q for j in range(N)]
        Pmn = []
        for i in range(N):
            acc = np.zeros((M, M))
            for j in range(N):
                if W[j, i] > 0:
                    acc += W[j, i] * APA[j]
            Pmn.append(0.5 * (acc + acc.T))
        d = max(np.max(np.abs(Pmn[i] - Pm[i])) for i in range(N))
        Pm = Pmn
        if d < tol:
            break
    return P, alpha


def collect_reachable(trials=120, seed=5, keep=160):
    rng = np.random.default_rng(seed)
    MM, alphas, traces = [], [], []
    for t in range(trials):
        r = rng.uniform(*R_RANGE, size=N)
        W = random_colstoch(rng) if t % 2 else iwm_from_sa(sa_matrix(rng))
        P, alpha = steady_state(W, r)
        for i in range(N):
            MM.append((np.eye(M) - P[i] @ (alpha[i] * LAM)) @ A)
        alphas += alpha.tolist()
        traces += [np.trace(p) for p in P]
    MM = np.array(MM)
    # keep the extremal blocks: largest spectral radius and largest norm
    rh = np.array([max(abs(np.linalg.eigvals(m))) for m in MM])
    nm = np.array([np.linalg.norm(m, 2) for m in MM])
    idx = np.unique(np.concatenate([np.argsort(rh)[-keep // 2:],
                                    np.argsort(nm)[-keep // 2:]]))
    return MM, MM[idx], (min(alphas), max(alphas)), (min(traces), max(traces)), rh


def weighted_norm(V, Mstack):
    """max_i || L^T M_i L^{-T} ||_2  where V = L L^T."""
    L = np.linalg.cholesky(V)
    Linv = np.linalg.inv(L)
    B = np.einsum('ji,kjl,ml->kim', L, Mstack, Linv)   # L^T M L^{-T}
    return float(max(np.linalg.norm(b, 2) for b in B))


def sym_from_vec(v):
    S = np.zeros((M, M))
    iu = np.triu_indices(M)
    S[iu] = v
    S = S + S.T - np.diag(np.diag(S))
    return S


def V_from_vec(v):
    w, U = np.linalg.eigh(sym_from_vec(v))
    w = np.clip(w, -12, 12)
    return U @ np.diag(np.exp(w)) @ U.T


def minimise_lambda(Mstack, iters=6000, seed=0):
    """Derivative-free search over V = expm(Sym) for the smallest common lambda."""
    rng = np.random.default_rng(seed)
    nv = M * (M + 1) // 2
    v = np.zeros(nv)
    best = weighted_norm(V_from_vec(v), Mstack)
    step = 0.6
    for it in range(iters):
        cand = v + step * rng.standard_normal(nv)
        try:
            f = weighted_norm(V_from_vec(cand), Mstack)
        except np.linalg.LinAlgError:
            continue
        if f < best:
            best, v = f, cand
        if (it + 1) % 800 == 0:
            step *= 0.6
    return V_from_vec(v), best


if __name__ == "__main__":
    print("=" * 74)
    MM_all, MM, arange, trange, rh = collect_reachable()
    print("reachable blocks sampled: %d  (extremal subset kept: %d)" % (len(MM_all), len(MM)))
    print("  alpha_i in [%.4f, %.4f]   predicted [1/r_max, 1/r_min] = [%.4f, %.4f]"
          % (arange[0], arange[1], 1 / R_RANGE[1], 1 / R_RANGE[0]))
    print("  trace P_i in [%.4f, %.4f]" % trange)
    print("  rho(M_i)  in [%.5f, %.5f]   <- any common lambda must exceed the max"
          % (rh.min(), rh.max()))

    print("=" * 74)
    print("searching for a common V ...")
    V, lam = minimise_lambda(MM)
    print("  best common lambda over the extremal subset : %.5f" % lam)
    lam_all = weighted_norm(V, MM_all)
    print("  the same V over ALL %d sampled blocks       : %.5f" % (len(MM_all), lam_all))
    print("  cond(V) = %.2f" % np.linalg.cond(V))
    if lam_all < 1.0:
        print("  => common quadratic Lyapunov function FOUND; uniform")
        print("     exponential stability holds for arbitrary switching with")
        print("     rate lambda = %.5f" % lam_all)
        np.save("common_V.npy", V)
        np.save("common_lambda.npy", np.array([lam_all]))
    else:
        print("  => no common V found with lambda < 1 in this search")
