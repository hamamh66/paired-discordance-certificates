"""Discordance-gated paired certificate (DGPC) for frozen diagnostic policies.

Reproduces every number and figure of the manuscript from the frozen audit
predictions in data/ (exported by notebooks/diagnostic_model_comparison.ipynb). CPU only.
Run:  python code/paired_certificate.py
"""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
FIG = ROOT.parent / "figures"
ALPHA, DELTA, SEED = 0.05, 0.01, 20261017
PI = np.linspace(.02, .80, 40)
T = np.linspace(.05, .60, 40)
P, W = np.meshgrid(PI, T / (1 - T), indexing="ij")
MODELS = ["Logistic", "Forest", "MLP"]
COLS = {"Logistic": "Logistic_action", "Forest": "Forest_action", "MLP": "Deep MLP_action"}
PAIRS = [(0, 1), (0, 2), (1, 2)]
N_CP = 2 * 2 * len(PAIRS) + 2 * len(MODELS)  # Bonferroni units: 18


def cp(k, n, a):
    """Two-sided exact Clopper-Pearson interval at level 1-a."""
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def paired_interval(xa, xb, a):
    """Interval for E[xa-xb] by discordance decomposition; level 1-2a."""
    n = len(xa)
    n10 = int(np.sum(xa & ~xb)); n01 = int(np.sum(~xa & xb)); k = n10 + n01
    dL, dU = cp(k, n, a)
    tL, tU = cp(n10, k, a)
    lo = min(dL * (2 * tL - 1), dU * (2 * tL - 1))
    hi = max(dL * (2 * tU - 1), dU * (2 * tU - 1))
    return lo, hi, (n10 - n01) / n, k


def dgpc(y, A):
    """Simultaneous (1-ALPHA) intervals for all paired and marginal rates."""
    a = ALPHA / N_CP
    out = {"pair": {}, "marg": {}}
    for c in (1, 0):
        Ac = A[y == c]
        for i, j in PAIRS:
            out["pair"][(i, j, c)] = paired_interval(Ac[:, i], Ac[:, j], a)
        for m in range(A.shape[1]):
            out["marg"][(m, c)] = cp(int(Ac[:, m].sum()), len(Ac), a)
    return out


def strategy_diff_bounds(y, A, cert, i, j):
    """Bounds on (Delta1, Delta0) for strategy i minus j; 3=all, 4=none."""
    def rate_iv(m, c):
        if m == 3: return (1.0, 1.0)
        if m == 4: return (0.0, 0.0)
        return cert["marg"][(m, c)]
    res = []
    for c in (1, 0):
        if i < 3 and j < 3:
            if (i, j, c) in cert["pair"]:
                lo, hi = cert["pair"][(i, j, c)][:2]
            else:
                lo, hi = cert["pair"][(j, i, c)][:2]; lo, hi = -hi, -lo
        else:
            li, ui = rate_iv(i, c); lj, uj = rate_iv(j, c)
            lo, hi = li - uj, ui - lj
        res.append((lo, hi))
    return res  # [(l1,u1),(l0,u0)]


def nb_lower(b):
    (l1, u1), (l0, u0) = b
    return P * l1 - (1 - P) * W * u0


def dkw_lower_pair(y, A, i, j):
    m = A.shape[1]; n1, n0 = (y == 1).sum(), (y == 0).sum()
    e1 = min(1, np.sqrt(np.log(4 * m / ALPHA) / (2 * n1)))
    e0 = min(1, np.sqrt(np.log(4 * m / ALPHA) / (2 * n0)))
    t = A[y == 1].mean(0); f = A[y == 0].mean(0)
    def lo(k): return P * max(0, t[k] - e1) - (1 - P) * W * min(1, f[k] + e0)
    def hi(k): return P * min(1, t[k] + e1) - (1 - P) * W * max(0, f[k] - e0)
    return lo(i) - hi(j), e1, e0


def winners(lower_fn):
    win = np.full(P.shape, -1)
    for i in range(5):
        ok = np.ones(P.shape, bool)
        for j in range(5):
            if j != i: ok &= lower_fn(i, j) > DELTA
        win[ok] = i
    return win


def analyse(name):
    df = pd.read_csv(ROOT.parent / "data" / f"{name}_audit_predictions.csv")
    y = df.y.to_numpy(); A = np.column_stack([df[COLS[m]].astype(bool).to_numpy() for m in MODELS])
    cert = dgpc(y, A)
    t = A[y == 1].mean(0); f = A[y == 0].mean(0)
    allnb = P - (1 - P) * W
    def nb_point(k): return allnb if k == 3 else (np.zeros_like(P) if k == 4 else P * t[k] - (1 - P) * W * f[k])
    def dkw_lower(i, j):
        if i < 3 and j < 3: return dkw_lower_pair(y, A, i, j)[0]
        n1, n0 = (y == 1).sum(), (y == 0).sum()
        e1 = min(1, np.sqrt(np.log(12 / ALPHA) / (2 * n1))); e0 = min(1, np.sqrt(np.log(12 / ALPHA) / (2 * n0)))
        def lo(k): return nb_point(k) if k > 2 else P * max(0, t[k] - e1) - (1 - P) * W * min(1, f[k] + e0)
        def hi(k): return nb_point(k) if k > 2 else P * min(1, t[k] + e1) - (1 - P) * W * max(0, f[k] - e0)
        return lo(i) - hi(j)
    def dgpc_lower(i, j): return nb_lower(strategy_diff_bounds(y, A, cert, i, j))
    w_dkw, w_dg = winners(dkw_lower), winners(dgpc_lower)
    rows = []
    for (i, j) in PAIRS:
        for c in (1, 0):
            lo, hi, est, k = cert["pair"][(i, j, c)]
            nc = int((y == c).sum())
            _, e1, e0 = dkw_lower_pair(y, A, i, j)
            e = e1 if c == 1 else e0
            rows.append(dict(dataset=name, pair=f"{MODELS[i]}-{MODELS[j]}", outcome="cases" if c else "controls",
                             n=nc, discordant=k, estimate=est, DGPC_lower=lo, DGPC_upper=hi,
                             DGPC_width=hi - lo, DKW_width=min(2, 4 * e)))
    # pairwise supported fraction and reserve over every ordered learned pair
    pair_rows = []
    for i in range(3):
        for j in range(3):
            if i == j: continue
            for meth, fn in (("DKW", dkw_lower), ("DGPC", dgpc_lower)):
                L = fn(i, j); r = np.clip((L - DELTA) / (2 * (P + (1 - P) * W)), 0, 1)
                pair_rows.append(dict(dataset=name, a=MODELS[i], b=MODELS[j], method=meth,
                                      supported_fraction=float((L > DELTA).mean()), max_reserve=float(r.max())))
    LR_M = dict(DKW=dkw_lower(0, 2), DGPC=dgpc_lower(0, 2), point=nb_point(0) - nb_point(2))
    return dict(name=name, y=y, A=A, rows=rows, pair_rows=pair_rows, w_dkw=w_dkw, w_dg=w_dg, LR_M=LR_M,
                sha=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(), n1=int(y.sum()), n0=int((1 - y).sum()))


def simulate(rng, reps=2000):
    """Coverage and certification behaviour on known joint action laws."""
    # joint states (a,b): 00,01,10,11
    laws = {"cases": np.array([.10, .04, .12, .74]), "controls": np.array([.80, .06, .02, .12])}
    true = {c: p[2] - p[1] for c, p in laws.items()}  # Delta = P(10)-P(01)
    states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], bool)
    true_L = P * true["cases"] - (1 - P) * W * true["controls"]
    out = []
    for n in (25, 50, 200, 1000):
        a = ALPHA / 4  # one pair, two classes, two CP each
        cov = 0; fals = {"DGPC": 0, "DKW": 0}; frac = {"DGPC": [], "DKW": []}
        e = np.sqrt(np.log(8 / ALPHA) / (2 * n))  # DKW with M=2
        for _ in range(reps):
            b = {}; bd = {}
            for c, p in laws.items():
                S = states[rng.choice(4, n, p=p)]
                lo, hi, est, _ = paired_interval(S[:, 0], S[:, 1], a)
                b[c] = (lo, hi); cov_c = lo <= true[c] <= hi
                bd[c] = (est - 2 * e, est + 2 * e)
                b[c + "_ok"] = cov_c
            cov += int(b["cases_ok"] and b["controls_ok"])
            for meth, bb in (("DGPC", b), ("DKW", bd)):
                L = P * bb["cases"][0] - (1 - P) * W * bb["controls"][1]
                claim = L > DELTA
                fals[meth] += int(np.any(claim & (true_L <= DELTA)))
                frac[meth].append(claim.mean())
        out.append(dict(n_per_class=n, reps=reps, DGPC_joint_coverage=cov / reps,
                        DGPC_any_false_claim=fals["DGPC"] / reps, DKW_any_false_claim=fals["DKW"] / reps,
                        DGPC_mean_certified_fraction=float(np.mean(frac["DGPC"])),
                        DKW_mean_certified_fraction=float(np.mean(frac["DKW"])),
                        truly_material_fraction=float((true_L > DELTA).mean())))
    return out


def planning():
    """Plug-in planning: n per class so that the DGPC half-width of a paired
    difference with discordance d and balanced discordant split is <= eta.
    Uses expected counts; a design heuristic, not a guarantee."""
    rows = []; a = ALPHA / N_CP
    for d in (.05, .10, .20, .40):
        for eta in (.02, .05):
            n = 10
            while True:
                k = int(round(d * n)); lo, hi, _, _ = paired_interval(
                    np.r_[np.ones(k // 2, bool), np.zeros(k - k // 2, bool), np.zeros(n - k, bool)],
                    np.r_[np.zeros(k // 2, bool), np.ones(k - k // 2, bool), np.zeros(n - k, bool)], a)
                if (hi - lo) / 2 <= eta: break
                n = int(n * 1.02) + 1
            rows.append(dict(discordance=d, halfwidth=eta, n_per_class=n))
    for eta in (.02, .05):  # DKW pairwise half-width 2*eps with M=3
        rows.append(dict(discordance="DKW (any)", halfwidth=eta,
                         n_per_class=int(np.ceil(2 * np.log(12 / ALPHA) / eta ** 2))))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    res = [analyse("breast_cancer"), analyse("heart")]
    # ---- invariant checks
    for r in res:
        for row in r["rows"]:
            assert row["DGPC_lower"] <= row["estimate"] <= row["DGPC_upper"]
    # closed-form frontier equals grid minimum test on a toy bound
    b = [(0.05, 0.10), (-0.04, 0.02)]
    Lt = nb_lower(b); w_star = (P * 0.05 - DELTA) / ((1 - P) * 0.02)
    assert np.all((Lt > DELTA) == (W < w_star))
    sim = simulate(rng); plan = planning()
    # ---- figures
    lab = ["Logistic", "Forest", "MLP", "Act on all", "Act on none"]
    cm = ListedColormap(['#d9d9d9', '#D55E00', '#009E73', '#0072B2', '#CC79A7', '#F0E442'])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6), layout="constrained")
    for row, r in enumerate(res):
        for col, (key, ttl) in enumerate((("w_dkw", "DKW marginal bounds"), ("w_dg", "Paired certificate (DGPC)"))):
            im = ax[row, col].pcolormesh(T, PI, r[key], cmap=cm, vmin=-1.5, vmax=4.5, shading="nearest")
            ax[row, col].set_title(f"{r['name'].replace('_', ' ').title()}: {ttl}")
            ax[row, col].set_xlabel("Clinical threshold t"); ax[row, col].set_ylabel("Target prevalence")
            cb = fig.colorbar(im, ax=ax[row, col], ticks=range(-1, 5)); cb.ax.set_yticklabels(["Unresolved"] + lab)
    fig.savefig(FIG / "winner_maps.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for a_, r in zip(ax, res):
        for key, ls, c in (("point", "-", "k"), ("DGPC", "--", "#0072B2"), ("DKW", ":", "#D55E00")):
            a_.plot(T, r["LR_M"][key][np.argmin(abs(PI - .2))], ls, color=c, label=key)
        a_.axhline(DELTA, color="grey", lw=.8); a_.set_title(r["name"].replace("_", " ").title() + r", $\pi=0.20$")
        a_.set_xlabel("Clinical threshold t"); a_.set_ylabel("NB(Logistic) $-$ NB(MLP) / lower bound")
    ax[0].legend(); fig.savefig(FIG / "lr_mlp_slice.pdf"); plt.close(fig)
    out = dict(alpha=ALPHA, delta=DELTA, bonferroni_units=N_CP,
               datasets={r["name"]: dict(n1=r["n1"], n0=r["n0"], sha256=r["sha"],
                         DKW_unresolved=float((r["w_dkw"] < 0).mean()), DGPC_unresolved=float((r["w_dg"] < 0).mean()),
                         DGPC_winner_fractions={lab[k]: float((r["w_dg"] == k).mean()) for k in range(5)},
                         DKW_winner_fractions={lab[k]: float((r["w_dkw"] == k).mean()) for k in range(5)})
                         for r in res},
               paired_intervals=[x for r in res for x in r["rows"]],
               pairwise=[x for r in res for x in r["pair_rows"]], simulation=sim, planning=plan)
    (ROOT.parent / "results" / "dgpc_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("datasets", "simulation", "planning")}, indent=1, default=float))
    print(pd.DataFrame(out["paired_intervals"]).round(3).to_string())
    print(pd.DataFrame(out["pairwise"]).round(4).to_string())


if __name__ == "__main__":
    main()
