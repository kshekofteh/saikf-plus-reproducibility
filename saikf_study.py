"""
Numerical validation study for SAIKF+ (Situation-Assessment-driven Interactive
Kalman Filter), prepared for the IEEE TAES submission.

Everything is self-contained and seeded: the network, the SA transition
probabilities and the measurements are synthetic, so every number quoted in
the manuscript is reproducible from this file alone.

Experiments
  E1  Monte-Carlo validation of the closed-form steady-state network MSD.
  E2  Uniform stability under a time-varying (switching) IWM, and a direct
      numerical check of the uniform bound rho(Adot) <= rho(M0).
  E3  Effect of the self-weight beta on convergence rate and steady-state MSD.
  E4  SA-driven IWM versus distance-only IWM in a diverging-neighbour geometry.
  E5  Covariance-propagation rule: the w^2 form versus the convex (w) form.

Output: .dat files (pgfplots) + numbers.tex (macros) in ../manuscript/data/
Run time: a few minutes.
"""

import os
import json
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manuscript", "data")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Model: planar constant-velocity kinematics, position-only measurements
# ----------------------------------------------------------------------------
DT = 0.1
M = 4                                     # state dimension [x, y, vx, vy]
A = np.array([[1, 0, DT, 0],
              [0, 1, 0, DT],
              [0, 0, 1,  0],
              [0, 0, 0,  1]], dtype=float)
G = np.array([[DT**2 / 2, 0],
              [0, DT**2 / 2],
              [DT, 0],
              [0, DT]], dtype=float)
SIGMA_A = 0.5                             # acceleration noise std  [m/s^2]
Q = SIGMA_A**2 * (G @ G.T)
C1 = np.array([[1.0, 0, 0, 0],
               [0, 1.0, 0, 0]])
N = 8                                     # vehicles in the neighbourhood
SIGMA_R = np.array([0.8, 1.2, 1.0, 1.5, 0.9, 1.3, 1.1, 1.4])   # [m]
R_LIST = [s**2 * np.eye(2) for s in SIGMA_R]
C_LIST = [C1.copy() for _ in range(N)]
NCLUSTER = 5                              # MHG clusters reachable in the scene
BETA = 0.2

# 'conv' : P^-_i = sum_j w_ji A P_j A^T + Q      (corrected, used throughout)
# 'sq'   : P^-_i = sum_j w_ji^2 (A P_j A^T + Q)  (form as published)
MODE_DEFAULT = 'conv'


# ----------------------------------------------------------------------------
# IWM construction
# ----------------------------------------------------------------------------
def sa_transition_matrix(rng, n=N, m=NCLUSTER, support=3):
    """Row i = transition probabilities P_{V_i,C_k} of vehicle i over clusters."""
    Pr = np.zeros((n, m))
    for i in range(n):
        k = rng.integers(1, support + 1)
        cols = rng.choice(m, size=k, replace=False)
        w = rng.random(k) + 0.2
        Pr[i, cols] = w / w.sum()
    return Pr


def iwm_from_sa(Pr, beta, neighbours=None):
    """Column-stochastic IWM, with the degenerate-case renormalisation: the
    probability mass of a cluster that no admissible neighbour can reach is
    returned to the self weight.  Returns W with W[j,i] = e_{V_j,V_i}."""
    n, m = Pr.shape
    W = np.zeros((n, n))
    for i in range(n):
        nb = [j for j in range(n) if j != i] if neighbours is None else \
             [j for j in neighbours[i] if j != i]
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


def iwm_from_distance(pos, beta, delta):
    """Distance-only IWM of the baseline IKF: entries proportional to 1/d
    inside radius delta, columns normalised to one."""
    n = pos.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        nb = [j for j in range(n) if j != i and d[j] <= delta]
        if not nb:
            W[i, i] = 1.0
            continue
        inv = np.array([1.0 / max(d[j], 1e-6) for j in nb])
        inv = inv / inv.sum()
        for j, v in zip(nb, inv):
            W[j, i] = (1.0 - beta) * v
        W[i, i] = beta
    return W


# ----------------------------------------------------------------------------
# Deterministic recursions
# ----------------------------------------------------------------------------
def node_S(W, i):
    S = np.zeros((M, M))
    for j in range(N):
        if W[j, i] > 0:
            S += W[j, i] * C_LIST[j].T @ np.linalg.solve(R_LIST[j], C_LIST[j])
    return S


def predict_cov(W, P, mode):
    """Prior-covariance propagation.

      'sq'   : sum_j w_ji^2 (A P_j A^T + Q)   -- as published; valid only if the
               node errors are mutually uncorrelated, i.e. when the interaction
               cross-covariances are propagated separately.
      'mix'  : sum_j w_ji A P_j A^T + sum_j w_ji^2 Q  -- correlated node errors,
               independent per-node process noise.
      'conv' : sum_j w_ji (A P_j A^T + Q)     -- fully convex.
    """
    out = []
    for i in range(N):
        acc = np.zeros((M, M))
        qac = np.zeros((M, M))
        for j in range(N):
            if W[j, i] > 0:
                if mode == 'sq':
                    acc += W[j, i]**2 * (A @ P[j] @ A.T + Q)
                elif mode == 'mix':
                    acc += W[j, i] * (A @ P[j] @ A.T)
                    qac += W[j, i]**2 * Q
                else:
                    acc += W[j, i] * (A @ P[j] @ A.T + Q)
        out.append(0.5 * (acc + qac + (acc + qac).T))
    return out


def steady_state_P(W, mode=MODE_DEFAULT, iters=20000, tol=1e-14):
    """Fixed point of  P_i = [(P_i^-)^-1 + S_i]^-1 ,  P_i^- = predict_cov."""
    S = [node_S(W, i) for i in range(N)]
    Pm = [np.eye(M) * 10.0 for _ in range(N)]
    P = [np.eye(M) * 10.0 for _ in range(N)]
    for _ in range(iters):
        P = [0.5 * (p + p.T) for p in
             [np.linalg.inv(np.linalg.inv(Pm[i]) + S[i]) for i in range(N)]]
        Pmn = predict_cov(W, P, mode)
        d = max(np.max(np.abs(Pmn[i] - Pm[i])) for i in range(N))
        Pm = Pmn
        if d < tol:
            break
    return P, Pm, S


def global_matrices(W, P, S):
    """Adot = W^T (I - P S) W^T (I_N kron A), together with Gdot and Ddot."""
    Wt = np.kron(W.T, np.eye(M))
    Pg = np.zeros((N * M, N * M))
    Sg = np.zeros((N * M, N * M))
    for i in range(N):
        sl = slice(i * M, (i + 1) * M)
        Pg[sl, sl] = P[i]
        Sg[sl, sl] = S[i]
    IPS = np.eye(N * M) - Pg @ Sg
    A_dot = Wt @ IPS @ Wt @ np.kron(np.eye(N), A)
    G_dot = Wt @ IPS
    Dblk = np.zeros((N * M, N * 2))
    for i in range(N):
        for j in range(N):
            if W[j, i] > 0:
                Dblk[i * M:(i + 1) * M, j * 2:(j + 1) * 2] = \
                    W[j, i] * P[i] @ C_LIST[j].T @ np.linalg.inv(R_LIST[j])
    return A_dot, G_dot, Wt @ Dblk


R_GLOB = np.zeros((N * 2, N * 2))
for _j in range(N):
    R_GLOB[_j * 2:(_j + 1) * 2, _j * 2:(_j + 1) * 2] = R_LIST[_j]
Q_GLOB = np.kron(np.eye(N), Q)             # independent per-node process noise


def theoretical_msd(W, mode=MODE_DEFAULT):
    """Closed-form steady-state MSD from the Lyapunov equation."""
    P, Pm, S = steady_state_P(W, mode)
    A_dot, G_dot, D_dot = global_matrices(W, P, S)
    rho = max(abs(np.linalg.eigvals(A_dot)))
    Wt = np.kron(W.T, np.eye(M))
    GQ = G_dot @ Wt                        # noise enters through the plant's
    B = GQ @ Q_GLOB @ GQ.T + D_dot @ R_GLOB @ D_dot.T   # own interactive mixing
    n = N * M
    if rho >= 1 - 1e-9:
        return dict(P=P, S=S, A_dot=A_dot, rho=rho, msd_i=np.full(N, np.inf),
                    msd_ave=np.inf, Px=None)
    Px = np.linalg.solve(np.eye(n * n) - np.kron(A_dot, A_dot),
                         B.reshape(-1, order='F')).reshape(n, n, order='F')
    msd_i = np.array([np.trace(Px[i * M:(i + 1) * M, i * M:(i + 1) * M])
                      for i in range(N)])
    return dict(P=P, S=S, A_dot=A_dot, rho=rho, Px=Px, msd_i=msd_i,
                msd_ave=float(np.trace(Px) / N))


# ----------------------------------------------------------------------------
# Monte-Carlo run of the filter
# ----------------------------------------------------------------------------
LQ = np.linalg.cholesky(Q + 1e-14 * np.eye(M))


def run_filter(W_seq, T, rng, mode=MODE_DEFAULT, W_plant=None, x0=None):
    """One realisation of the interactive plant and the SAIKF+ network.

    Plant (the interactive model of the IKF framework):
        x_{i,t+1} = sum_j w*_{ji,t} ( A x_{j,t} + omega_{j,t} ),
    so each target's next state is the weighted combination of its neighbours'
    propagated states.  Node i estimates its own state x_i from its own
    measurement and those of its neighbours.

    `W_plant` is the plant's true coupling; when omitted the filter's own IWM
    is used, i.e. the filter's interaction model is correct.
    Returns sq[t,i] = ||x_{i,t} - xhat_{i,t}||^2.
    """
    X = np.tile(np.array([0.0, 0.0, 12.0, 3.0]), (N, 1)) if x0 is None else x0.copy()
    X = X + rng.standard_normal((N, M)) * np.array([2.0, 2.0, 0.5, 0.5])
    xh = [np.zeros(M) for _ in range(N)]
    Pm = [np.eye(M) * 10.0 for _ in range(N)]
    sq = np.zeros((T, N))
    Rinv = [np.linalg.inv(R) for R in R_LIST]
    for t in range(T):
        W = W_seq(t)
        Wp = W if W_plant is None else W_plant
        y = [C_LIST[j] @ X[j] + SIGMA_R[j] * rng.standard_normal(2) for j in range(N)]
        P, psi = [], []
        for i in range(N):
            S = np.zeros((M, M))
            q = np.zeros(M)
            for j in range(N):
                if W[j, i] > 0:
                    S += W[j, i] * C_LIST[j].T @ Rinv[j] @ C_LIST[j]
                    q += W[j, i] * C_LIST[j].T @ Rinv[j] @ y[j]
            Pi = np.linalg.inv(np.linalg.inv(Pm[i]) + S)
            Pi = 0.5 * (Pi + Pi.T)
            P.append(Pi)
            psi.append(xh[i] + Pi @ (q - S @ xh[i]))
        xh = [sum(W[j, i] * psi[j] for j in range(N) if W[j, i] > 0)
              for i in range(N)]
        for i in range(N):
            sq[t, i] = float(np.sum((X[i] - xh[i])**2))
        xh = [sum(W[j, i] * (A @ xh[j]) for j in range(N) if W[j, i] > 0)
              for i in range(N)]
        Pm = predict_cov(W, P, mode)
        prop = np.array([A @ X[j] + LQ @ rng.standard_normal(M) for j in range(N)])
        X = np.array([sum(Wp[j, i] * prop[j] for j in range(N) if Wp[j, i] > 0)
                      for i in range(N)])
    return sq


def mc_average(W_seq, T, reps, seed0, mode=MODE_DEFAULT, runner=None, W_plant=None):
    runner = run_filter if runner is None else runner
    acc = np.zeros((T, N))
    tails = np.zeros(reps)
    for r in range(reps):
        sq = runner(W_seq, T, np.random.default_rng(seed0 + r), mode, W_plant)
        acc += sq
        tails[r] = sq[-T // 3:].mean()
    return acc / reps, tails


# ============================================================================
res = {}
rng0 = np.random.default_rng(20261008)
Pr = sa_transition_matrix(rng0)
W_sa = iwm_from_sa(Pr, BETA)
assert np.allclose(W_sa.sum(axis=0), 1.0), "IWM is not column-stochastic"
res['colsum_err'] = float(np.max(np.abs(W_sa.sum(axis=0) - 1.0)))

# ---------------- E1 : closed-form MSD versus Monte Carlo ------------------
T, MC = 300, 500
th = theoretical_msd(W_sa)
emp, tails = mc_average(lambda t: W_sa, T, MC, 1000)
tail_node = emp[-100:].mean(axis=0)

res.update(rho_sa=float(th['rho']), msd_th=float(th['msd_ave']),
           msd_emp=float(tail_node.mean()), mc=MC, T=T,
           msd_err_pct=float(100 * abs(tail_node.mean() - th['msd_ave']) / th['msd_ave']),
           msd_node_max_err_pct=float(100 * np.max(np.abs(tail_node - th['msd_i']) / th['msd_i'])),
           msd_ci=float(1.96 * tails.std(ddof=1) / np.sqrt(MC)))
print("E1 rho=%.4f  theory=%.4f  empirical=%.4f (%.2f%%)  per-node max err %.2f%%"
      % (res['rho_sa'], res['msd_th'], res['msd_emp'], res['msd_err_pct'],
         res['msd_node_max_err_pct']))

with open(os.path.join(OUT, "e1_msd_curve.dat"), "w") as f:
    f.write("t emp th\n")
    for t in range(T):
        f.write("%d %.6g %.6g\n" % (t, emp[t].mean(), th['msd_ave']))
with open(os.path.join(OUT, "e1_msd_per_node.dat"), "w") as f:
    f.write("node emp th sigma\n")
    for i in range(N):
        f.write("%d %.6g %.6g %.3g\n" % (i + 1, tail_node[i], th['msd_i'][i], SIGMA_R[i]))

# ---------------- E2 : time-varying topology -------------------------------
MODES = 6
W_modes = [iwm_from_sa(sa_transition_matrix(np.random.default_rng(7000 + k)), BETA)
           for k in range(MODES)]
th_modes = [theoretical_msd(Wm) for Wm in W_modes]
rho_modes = [t['rho'] for t in th_modes]
Ad_modes = [t['A_dot'] for t in th_modes]
rho_Mi = [max(abs(np.linalg.eigvals((np.eye(M) - t['P'][i] @ t['S'][i]) @ A)))
          for t in th_modes + [th] for i in range(N)]
res.update(n_modes=MODES, rho_modes_max=float(max(rho_modes)),
           rho_modes_min=float(min(rho_modes)), rho_Mi_max=float(max(rho_Mi)),
           msd_modes_worst=float(max(t['msd_ave'] for t in th_modes)))

sw_rng = np.random.default_rng(31337)
seq = sw_rng.integers(0, MODES, size=T)
prod = np.eye(N * M)
norms = []
for t in range(150):
    prod = Ad_modes[seq[t]] @ prod
    norms.append(float(np.linalg.norm(prod, 2)))
jsr = max(n**(1.0 / (k + 1)) for k, n in enumerate(norms[20:], start=20))
res['jsr_est'] = float(jsr)

emp_sw, tails_sw = mc_average(lambda t: W_modes[seq[t]], T, 200, 5000)
res.update(msd_switch=float(emp_sw[-100:].mean()),
           msd_switch_ci=float(1.96 * tails_sw.std(ddof=1) / np.sqrt(200)))
print("E2 modes rho in [%.4f, %.4f]  JSR est %.4f  switching MSD %.4f  worst fixed %.4f"
      % (res['rho_modes_min'], res['rho_modes_max'], res['jsr_est'],
         res['msd_switch'], res['msd_modes_worst']))

with open(os.path.join(OUT, "e2_switching.dat"), "w") as f:
    f.write("t sw bound\n")
    for t in range(T):
        f.write("%d %.6g %.6g\n" % (t, emp_sw[t].mean(), res['msd_modes_worst']))
with open(os.path.join(OUT, "e2_product_norm.dat"), "w") as f:
    f.write("t norm bound\n")
    for t, v in enumerate(norms):
        f.write("%d %.6g %.6g\n" % (t + 1, v, res['rho_Mi_max']**(t + 1)))

# E2b : the uniform bound of Theorem 3 for homogeneous nodes
R_HET = list(R_LIST)
R_LIST = [1.0 * np.eye(2) for _ in range(N)]      # homogeneous nodes
hom_rng = np.random.default_rng(555)
viol, ratios = 0, []
for trial in range(200):
    Wt_ = iwm_from_sa(sa_transition_matrix(hom_rng), BETA)
    P_, _, S_ = steady_state_P(Wt_)
    Ad_, _, _ = global_matrices(Wt_, P_, S_)
    r_ad = max(abs(np.linalg.eigvals(Ad_)))
    r_m0 = max(abs(np.linalg.eigvals((np.eye(M) - P_[0] @ S_[0]) @ A)))
    ratios.append(r_ad / r_m0)
    if r_ad > r_m0 + 1e-8:
        viol += 1
res.update(hom_trials=200, hom_viol=viol, hom_ratio_max=float(max(ratios)))
R_LIST = R_HET
print("E2b homogeneous check: %d/200 violations of rho(Adot)<=rho(M0), max ratio %.4f"
      % (viol, res['hom_ratio_max']))

# ---------------- E3 : beta sweep ------------------------------------------
betas = np.round(np.arange(0.0, 0.96, 0.05), 3)
rows = [(b,) + tuple(np.array([theoretical_msd(iwm_from_sa(Pr, b))[k]
                               for k in ('rho', 'msd_ave')])) for b in betas]
best = min(rows, key=lambda r: r[2])
pick = lambda b: [r for r in rows if abs(r[0] - b) < 1e-9][0]
res.update(beta_best=float(best[0]), msd_beta_best=float(best[2]),
           msd_beta0=float(pick(0.0)[2]), msd_beta02=float(pick(0.2)[2]),
           msd_beta09=float(pick(0.9)[2]), rho_beta0=float(pick(0.0)[1]),
           rho_beta09=float(pick(0.9)[1]),
           msd_penalty_02=float(100 * (pick(0.2)[2] - best[2]) / best[2]))
print("E3 best beta %.2f (MSD %.4f); beta=0.2 -> %.4f ; beta=0.9 -> %.4f"
      % (res['beta_best'], res['msd_beta_best'], res['msd_beta02'], res['msd_beta09']))
with open(os.path.join(OUT, "e3_beta.dat"), "w") as f:
    f.write("beta rho msd\n")
    for b, r_, m_ in rows:
        f.write("%.3f %.6g %.6g\n" % (b, r_, m_))

# ---------------- E4 : SA-driven versus distance-only IWM ------------------
geo_rng = np.random.default_rng(424242)
pos = np.array([[0.0, 0.0], [6.0, 2.0], [4.0, -3.0], [-5.0, 4.0],
                [8.0, -6.0], [-7.0, -5.0], [9.0, 7.0], [-9.0, 1.0]])
Pr_div = np.zeros((N, NCLUSTER))
for i in [0, 1, 2, 3]:                    # same destination  -> clusters {0,1}
    w = geo_rng.random(2) + 0.3
    Pr_div[i, [0, 1]] = w / w.sum()
for i in [4, 5, 6, 7]:                    # diverging          -> clusters {3,4}
    w = geo_rng.random(2) + 0.3
    Pr_div[i, [3, 4]] = w / w.sum()

W_div_sa = iwm_from_sa(Pr_div, BETA)
W_div_d = iwm_from_distance(pos, BETA, delta=15.0)
th4_sa, th4_d = theoretical_msd(W_div_sa), theoretical_msd(W_div_d)

# The plant's true coupling is the SA-consistent one: the two flows are
# dynamically uncoupled.  The SA-driven filter models that correctly; the
# distance-only filter assumes a coupling the plant does not have.
MC4, T4 = 400, 300
e_sa, tails_sa = mc_average(lambda t: W_div_sa, T4, MC4, 90000, W_plant=W_div_sa)
e_d, tails_d = mc_average(lambda t: W_div_d, T4, MC4, 90000, W_plant=W_div_sa)
d = tails_d - tails_sa
res.update(e4_mc=MC4, e4_msd_sa=float(tails_sa.mean()), e4_msd_dist=float(tails_d.mean()),
           e4_std_sa=float(tails_sa.std(ddof=1)), e4_std_dist=float(tails_d.std(ddof=1)),
           e4_impr=float(100 * (tails_d.mean() - tails_sa.mean()) / tails_d.mean()),
           e4_t=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
           e4_rho_sa=float(th4_sa['rho']), e4_rho_dist=float(th4_d['rho']),
           e4_th_sa=float(th4_sa['msd_ave']), e4_th_dist=float(th4_d['msd_ave']),
           e4_win=float(100.0 * np.mean(d > 0)))
print("E4 SA %.4f+-%.4f   distance %.4f+-%.4f   improvement %.1f%%  t=%.1f  wins %.0f%%"
      % (res['e4_msd_sa'], res['e4_std_sa'], res['e4_msd_dist'],
         res['e4_std_dist'], res['e4_impr'], res['e4_t'], res['e4_win']))
with open(os.path.join(OUT, "e4_sa_vs_dist.dat"), "w") as f:
    f.write("t sa dist\n")
    for t in range(T4):
        f.write("%d %.6g %.6g\n" % (t, e_sa[t].mean(), e_d[t].mean()))

# ---------------- E5 : covariance-propagation rule -------------------------
for tag, md in (('sq', 'sq'), ('mix', 'mix')):
    thm = theoretical_msd(W_sa, mode=md)
    Pm_, _, _ = steady_state_P(W_sa, mode=md)
    empm, _ = mc_average(lambda t: W_sa, T, 80, 2000, mode=md)
    res['e5_rho_' + tag] = float(thm['rho'])
    res['e5_trP_' + tag] = float(np.mean([np.trace(p) for p in Pm_]))
    res['e5_msd_' + tag] = float(empm[-100:].mean())
    globals()['emp_' + tag] = empm
res.update(e5_rho_conv=float(th['rho']),
           e5_trP_conv=float(np.mean([np.trace(p) for p in th['P']])),
           e5_msd_conv=float(emp[-100:].mean()),
           e5_wsum_sq=float(np.min((W_sa**2).sum(axis=0))),
           e5_wsum_sq_max=float(np.max((W_sa**2).sum(axis=0))))
print("E5  sq: rho=%.6f trP=%.4f MSD=%.4g | mix: rho=%.4f trP=%.4f MSD=%.4g"
      " | conv: rho=%.4f trP=%.4f MSD=%.4g"
      % (res['e5_rho_sq'], res['e5_trP_sq'], res['e5_msd_sq'],
         res['e5_rho_mix'], res['e5_trP_mix'], res['e5_msd_mix'],
         res['e5_rho_conv'], res['e5_trP_conv'], res['e5_msd_conv']))
with open(os.path.join(OUT, "e5_propagation.dat"), "w") as f:
    f.write("t sq mix conv\n")
    for t in range(T):
        f.write("%d %.6g %.6g %.6g\n"
                % (t, emp_sq[t].mean(), emp_mix[t].mean(), emp[t].mean()))

# ---------------- macros ---------------------------------------------------
FMT = dict(numMC=('mc', "%d"), numT=('T', "%d"), numModes=('n_modes', "%d"),
           rhoSA=('rho_sa', "%.4f"), msdTh=('msd_th', "%.4f"),
           msdEmp=('msd_emp', "%.4f"), msdErrPct=('msd_err_pct', "%.2f"),
           msdNodeErrPct=('msd_node_max_err_pct', "%.2f"), msdCI=('msd_ci', "%.4f"),
           rhoModesMax=('rho_modes_max', "%.4f"), rhoModesMin=('rho_modes_min', "%.4f"),
           rhoMiMax=('rho_Mi_max', "%.4f"), jsrEst=('jsr_est', "%.4f"),
           msdSwitch=('msd_switch', "%.4f"), msdModesWorst=('msd_modes_worst', "%.4f"),
           homTrials=('hom_trials', "%d"), homViol=('hom_viol', "%d"),
           homRatioMax=('hom_ratio_max', "%.4f"),
           betaBest=('beta_best', "%.2f"), msdBetaBest=('msd_beta_best', "%.4f"),
           msdBetaZero=('msd_beta0', "%.4f"), msdBetaTwo=('msd_beta02', "%.4f"),
           msdBetaNine=('msd_beta09', "%.4f"), rhoBetaZero=('rho_beta0', "%.4f"),
           rhoBetaNine=('rho_beta09', "%.4f"), msdPenaltyTwo=('msd_penalty_02', "%.1f"),
           eFourMC=('e4_mc', "%d"), eFourMsdSA=('e4_msd_sa', "%.4f"),
           eFourMsdDist=('e4_msd_dist', "%.4f"), eFourStdSA=('e4_std_sa', "%.4f"),
           eFourStdDist=('e4_std_dist', "%.4f"), eFourImpr=('e4_impr', "%.1f"),
           eFourT=('e4_t', "%.1f"), eFourWin=('e4_win', "%.0f"),
           eFourRhoSA=('e4_rho_sa', "%.4f"), eFourRhoDist=('e4_rho_dist', "%.4f"),
           eFiveRhoSq=('e5_rho_sq', "%.5f"), eFiveTrPSq=('e5_trP_sq', "%.4f"),
           eFiveRhoMix=('e5_rho_mix', "%.4f"), eFiveTrPMix=('e5_trP_mix', "%.4f"),
           eFiveMsdMix=('e5_msd_mix', "%.4f"), eFiveMsdConv=('e5_msd_conv', "%.4f"),
           eFiveTrPConv=('e5_trP_conv', "%.4f"), eFiveMsdSq=('e5_msd_sq', "%.3g"),
           eFiveWsumMin=('e5_wsum_sq', "%.3f"), eFiveWsumMax=('e5_wsum_sq_max', "%.3f"),
           numNodes=(None, "%d"))
with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write("%% auto-generated by numerical/saikf_study.py -- do not edit\n")
    f.write("\\newcommand{\\numNodes}{%d}\n" % N)
    for macro, (key, fmt) in FMT.items():
        if key is None:
            continue
        f.write("\\newcommand{\\%s}{%s}\n" % (macro, fmt % res[key]))
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(res, f, indent=2)
print("\nwrote", os.path.normpath(OUT))
