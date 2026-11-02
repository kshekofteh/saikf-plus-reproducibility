"""Generate the heterogeneity-sweep data for the TSP manuscript figure."""
import numpy as np, json, os
import common_lyapunov as cl
from verify_theorem import Adot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manuscript", "data")
os.makedirs(OUT, exist_ok=True)

def jsr_est(r_range, n_modes=6, horizon=200, runs=20, seed=4):
    rng = np.random.default_rng(seed)
    modes = [cl.iwm_from_sa(cl.sa_matrix(np.random.default_rng(900+k))) for k in range(n_modes)]
    r = rng.uniform(*r_range, size=cl.N)
    PS = [cl.steady_state(W, r) for W in modes]
    ops, best = {}, 0.0
    for _ in range(runs):
        seq = rng.integers(0, n_modes, size=horizon+1)
        prod = np.eye(cl.N*cl.M)
        for t in range(1, horizon+1):
            a, b = seq[t], seq[t-1]
            if (a,b) not in ops:
                ops[(a,b)] = Adot(modes[a], modes[b], PS[a][0], PS[a][1])
            prod = ops[(a,b)] @ prod
            if t >= 60:
                best = max(best, np.linalg.norm(prod,2)**(1.0/t))
    return best, max(max(abs(np.linalg.eigvals(o))) for o in ops.values())

rows = []
for spread in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 13.0, 17.0, 25.0):
    lo, hi = 1.0/np.sqrt(spread), np.sqrt(spread)
    cl.R_RANGE = (lo**2, hi**2)
    MM_all, MM, arange, trange, rh = cl.collect_reachable(trials=40, seed=5, keep=60)
    V, _ = cl.minimise_lambda(MM, iters=2500, seed=1)
    lam = cl.weighted_norm(V, MM_all)
    j, pm = jsr_est(cl.R_RANGE)
    rows.append((spread, lam, j, pm, rh.max(), arange[0], arange[1]))
    print("spread %-6.1f lambda %-9.4f jsr %-9.4f maxrho %-9.4f" % (spread, lam, j, rh.max()))

with open(os.path.join(OUT, "h1_spread.dat"), "w") as f:
    f.write("spread lambda jsr moderho rhomax alo ahi\n")
    for r_ in rows:
        f.write("%.3f %.6g %.6g %.6g %.6g %.6g %.6g\n" % r_)

res = dict(
    spread_app=1.875,
    lambda_app=float([r[1] for r in rows if r[0]==4.0][0]),
    lambda_homog=float([r[1] for r in rows if r[0]==1.0][0]),
    rho_homog=float([r[4] for r in rows if r[0]==1.0][0]),
    jsr_homog=float([r[2] for r in rows if r[0]==1.0][0]),
    cert_limit=max([r[0] for r in rows if r[1] < 1.0]),
    jsr_limit=max([r[0] for r in rows if r[2] < 1.0]),
    n_blocks=960, n_colstoch=40, wnorm=1.0,
)
json.dump(res, open(os.path.join(OUT,"h_results.json"),"w"), indent=2)
print("\n", json.dumps(res, indent=2))
