"""MF731 HW 9 (09_stress_testing_pub): K-day stress test of a cap-weighted MSFT/AAPL portfolio.

Index (1) = MSFT, (2) = AAPL. Risk-factor changes are daily log returns X.
  EWMA:  mu_{s+D} = lam mu_s + (1-lam) X_s,  Sigma_{s+D} = th Sigma_s + (1-th)(X_s-mu_s)(X_s-mu_s)'
  loss:  L^lin_{t+K D} = -theta_t' sum_k X_{t+k D},  theta_t = V w held fixed (constant shares).
Day 1 of every shocked path is the same deterministic two-stage 5-sigma shock; days 2..K
are sampled from N(mu, Sigma) with EWMA updates path by path.
Writes generated/values.tex, generated/table.tex and figures/losses.png.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.join(os.path.dirname(__file__), "..")
CAPS = np.array([448.77, 575.11])                 # $ billions on 9/1/16, from the sheet
V0, ALPHA, K, LAM, TH, M, SEED = 1_000_000.0, 0.95, 10, 0.97, 0.97, 50_000, 731
START, END = "2011-09-01", "2016-09-01"


def load_returns(name="stocks_close.csv"):
    """Frozen Yahoo closes (downloaded 2026-10-23). Prices 9/1/11..9/1/16, so the first
    return is 9/2/11 over 9/1/11 and the last is 9/1/16 over 8/31/16."""
    P = pd.read_csv(os.path.join(ROOT, "data", name), index_col=0, parse_dates=True)
    P = P.loc[START:END, ["MSFT", "AAPL"]]
    return np.log(P / P.shift(1)).dropna()


def ewma(X, lam=LAM, th=TH):
    """Run the sheet's recursion from mu = 0, Sigma = 0 through every return.
    The output is (mu_{t+D}, Sigma_{t+D}) with t = 9/1/16."""
    mu, S = np.zeros(2), np.zeros((2, 2))
    for x in X:
        dev = x - mu                               # uses mu_s, before its update
        mu = lam * mu + (1 - lam) * x
        S = th * S + (1 - th) * np.outer(dev, dev)
    return mu, S


def var_emp(l, alpha=ALPHA):
    """VaR_alpha(F_n) = l_(ceil(n alpha)), the lecture's order-statistic estimator."""
    l = np.sort(l)
    return l[int(np.ceil(len(l) * alpha - 1e-12)) - 1]


def ewma_step(mu, S, x):
    """Vectorized EWMA update over paths: mu (M,2), S (M,2,2), x (M,2)."""
    dev = x - mu
    return LAM * mu + (1 - LAM) * x, TH * S + (1 - TH) * dev[:, :, None] * dev[:, None, :]


def sample(mu, S, z):
    """Draw N(mu, S) path by path with the explicit 2x2 Cholesky factor."""
    a = np.sqrt(S[:, 0, 0])
    b = S[:, 0, 1] / a
    c = np.sqrt(S[:, 1, 1] - b**2)
    return mu + np.stack([a * z[:, 0], b * z[:, 0] + c * z[:, 1]], axis=1)


def shock(mu, S, k1=5.0, k2=5.0):
    """Step 3: x2 = mu2 - k2 sigma2, then x1 = conditional mean - k1 conditional sd."""
    s1, s2 = np.sqrt(np.diag(S))
    rho = S[0, 1] / (s1 * s2)
    x2 = mu[1] - k2 * s2
    mu1c = mu[0] + rho * s1 / s2 * (x2 - mu[1])
    s1c = s1 * np.sqrt(1 - rho**2)
    return np.array([mu1c - k1 * s1c, x2]), mu1c, s1c


def simulate(mu, S, theta, mode, rng, m=M):
    """K-day linearized losses on m paths.
    mode 'sheet'   : day-1 shock as in the sheet (x1 = mu1c - 5 s1c).
    mode 'lecture' : x2 shocked, x1 drawn from its conditional law (slides 43-45).
    mode 'none'    : no shock, day 1 drawn from N(mu, S)."""
    mu_p, S_p = np.tile(mu, (m, 1)), np.tile(S, (m, 1, 1))
    x1day, mu1c, s1c = shock(mu, S)
    if mode == "sheet":
        x = np.tile(x1day, (m, 1))
    elif mode == "lecture":
        x = np.column_stack([mu1c + s1c * rng.standard_normal(m), np.full(m, x1day[1])])
    else:
        x = sample(mu_p, S_p, rng.standard_normal((m, 2)))
    total = x.copy()
    rho_after = None
    for k in range(2, K + 1):
        mu_p, S_p = ewma_step(mu_p, S_p, x)       # mu_{t+k D}, Sigma_{t+k D}
        if k == 2:
            rho_after = S_p[:, 0, 1] / np.sqrt(S_p[:, 0, 0] * S_p[:, 1, 1])
        x = sample(mu_p, S_p, rng.standard_normal((m, 2)))
        total += x                                 # no update needed after day K
    return -(total @ theta), rho_after


def stats(l, var_K, reg):
    return dict(avg=l.mean(), var=var_emp(l), exc_var=100 * np.mean(l > var_K),
                exc_reg=100 * np.mean(l > reg), se_avg=l.std(ddof=1) / np.sqrt(len(l)))


def run(price_file="stocks_close.csv", seed=SEED):
    X = load_returns(price_file)
    w = CAPS / CAPS.sum()
    theta = V0 * w
    mu, S = ewma(X.values)
    s = np.sqrt(np.diag(S))
    rho = S[0, 1] / (s[0] * s[1])
    var1 = -theta @ mu + np.sqrt(theta @ S @ theta) * norm.ppf(ALPHA)   # step 2, exact
    varK = np.sqrt(K) * var1
    reg = 3 * varK
    x1day, mu1c, s1c = shock(mu, S)
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("sheet", "lecture", "none"):
        l, r2 = simulate(mu, S, theta, mode, rng)
        out[mode] = stats(l, varK, reg) | dict(l=l, rho2=r2)
    # Exact means: E[X_{t+kD} | F] = E[mu_{t+kD}], and E[mu] is constant after day 1
    # because E[mu_{k+1}] = lam E[mu_k] + (1-lam) E[X_k] = E[mu_k].
    ex1 = {"sheet": x1day, "lecture": np.array([mu1c, x1day[1]]), "none": mu}
    for mode, e1 in ex1.items():
        mu2 = LAM * mu + (1 - LAM) * e1
        out[mode]["exact_avg"] = -(theta @ e1) - (K - 1) * (theta @ mu2)
    var_iid = -K * theta @ mu + np.sqrt(K * theta @ S @ theta) * norm.ppf(ALPHA)
    return dict(X=X, w=w, theta=theta, mu=mu, S=S, s=s, rho=rho, var1=var1, varK=varK,
                reg=reg, x=x1day, mu1c=mu1c, s1c=s1c, out=out, var_iid=var_iid,
                day1_loss=-(theta @ x1day), mu2=LAM * mu + (1 - LAM) * x1day)


def checks(r):
    """Assertions that the pieces are what the write-up says they are."""
    mu, S, theta = r["mu"], r["S"], r["theta"]
    rng = np.random.default_rng(1)
    # exact one-day linear VaR vs Monte Carlo
    Z = rng.multivariate_normal(mu, S, size=2_000_000)
    mc1 = var_emp(-(Z @ theta))
    assert abs(mc1 / r["var1"] - 1) < 5e-3, (mc1, r["var1"])
    # conditional law of X1 | X2 = x2 by regression on the same draws
    sel = np.abs(Z[:, 1] - (mu[1] - 1.0 * r["s"][1])) < 0.02 * r["s"][1]
    _, mc_c, sc_c = shock(mu, S, 0.0, 1.0)
    assert abs(Z[sel, 0].mean() - mc_c) < 0.05 * r["s"][0]
    assert abs(Z[sel, 0].std() / sc_c - 1) < 0.05
    # vectorized sampler reproduces the covariance
    m = 400_000
    x = sample(np.zeros((m, 2)), np.tile(S, (m, 1, 1)), rng.standard_normal((m, 2)))
    assert np.allclose(np.cov(x.T), S, rtol=2e-2, atol=1e-7)
    for mode, o in r["out"].items():               # Monte Carlo mean vs exact mean
        assert abs(o["avg"] - o["exact_avg"]) < 3.5 * o["se_avg"], (mode, o["avg"], o["exact_avg"])
    # EWMA start does not matter: weight left on the zero initial value
    n = len(r["X"])
    return dict(mc1=mc1, w0=LAM**n, n=n)


def fmt(v, d=0):
    return f"{v:,.{d}f}"


def write_outputs(r, c, adj, seeds):
    o = r["out"]
    gen = os.path.join(ROOT, "generated")
    vals = {
        "Nret": str(len(r["X"])), "FirstRet": r["X"].index[0].strftime("%-m/%-d/%y"),
        "LastRet": r["X"].index[-1].strftime("%-m/%-d/%y"),
        "wOne": fmt(100 * r["w"][0], 2), "wTwo": fmt(100 * r["w"][1], 2),
        "thOne": fmt(r["theta"][0], 2), "thTwo": fmt(r["theta"][1], 2),
        "muOne": f"{r['mu'][0]:.3e}".replace("e-0", r"\times10^{-") + "}",
        "muTwo": f"{r['mu'][1]:.3e}".replace("e-0", r"\times10^{-") + "}",
        "sOne": fmt(100 * r["s"][0], 4), "sTwo": fmt(100 * r["s"][1], 4),
        "rhoT": fmt(r["rho"], 4),
        "VarOne": fmt(r["var1"], 2), "VarK": fmt(r["varK"], 2), "Reg": fmt(r["reg"], 2),
        "MCvar": fmt(c["mc1"], 0), "WzeroE": f"{c['w0']:.1e}",
        "xTwo": fmt(100 * r["x"][1], 4), "muOneC": fmt(100 * r["mu1c"], 4),
        "sOneC": fmt(100 * r["s1c"], 4), "xOne": fmt(100 * r["x"][0], 4),
        "DayOneLoss": fmt(r["day1_loss"], 2),
        "rhoTwo": fmt(o["sheet"]["rho2"][0], 4),
        "AdjVarOne": fmt(adj["var1"], 2), "AdjRho": fmt(adj["rho"], 4),
        "AdjAvg": fmt(adj["out"]["sheet"]["avg"]), "AdjVaR": fmt(adj["out"]["sheet"]["var"]),
        "AdjExcVar": fmt(adj["out"]["sheet"]["exc_var"], 2),
        "AdjExcReg": fmt(adj["out"]["sheet"]["exc_reg"], 2),
        "SeedLo": fmt(min(seeds), 2), "SeedHi": fmt(max(seeds), 2),
    }
    for mode, tag in (("sheet", "S"), ("lecture", "L"), ("none", "N")):
        vals[tag + "avg"] = fmt(o[mode]["avg"])
        vals[tag + "se"] = fmt(o[mode]["se_avg"])
        vals[tag + "var"] = fmt(o[mode]["var"])
        vals[tag + "excvar"] = fmt(o[mode]["exc_var"], 2)
        vals[tag + "excreg"] = fmt(o[mode]["exc_reg"], 3)
        vals[tag + "exact"] = fmt(o[mode]["exact_avg"])
        p_ = o[mode]["exc_var"] / 100
        vals[tag + "seexc"] = fmt(100 * np.sqrt(p_ * (1 - p_) / M), 2)
        p_ = o[mode]["exc_reg"] / 100
        vals[tag + "seexcreg"] = fmt(100 * np.sqrt(p_ * (1 - p_) / M), 2)
    vals["VarIID"] = fmt(r["var_iid"])
    d = r["x"] - r["mu"]
    S2 = TH * r["S"] + (1 - TH) * np.outer(d, d)
    vals["sOneTwo"] = fmt(100 * np.sqrt(S2[0, 0]), 2)
    vals["sTwoTwo"] = fmt(100 * np.sqrt(S2[1, 1]), 2)
    vals["sdRise"] = fmt(100 * (np.sqrt(r["theta"] @ S2 @ r["theta"] / (r["theta"] @ r["S"] @ r["theta"])) - 1))
    vals["RegRatio"] = fmt(o["sheet"]["exc_reg"] / o["lecture"]["exc_reg"], 1)
    vals["muTwoOne"] = fmt(100 * r["mu2"][0], 4)
    vals["muTwoTwo"] = fmt(100 * r["mu2"][1], 4)
    with open(os.path.join(gen, "values.tex"), "w") as f:
        for k, v in vals.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    rows = [("Average 10-day loss (\\$)", "avg", 0), ("10-day VaR$_{0.95}$ (\\$)", "var", 0),
            ("\\% losses $>\\sqrt{10}\\,\\widehat{\\VaR}$", "exc_var", 2),
            ("\\% losses $>3\\sqrt{10}\\,\\widehat{\\VaR}$", "exc_reg", 3)]
    with open(os.path.join(gen, "table.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n & Sheet shock & Lecture shock & No shock\\\\\n\\midrule\n")
        for lab, key, d in rows:
            f.write(f"{lab} & \\textbf{{{fmt(o['sheet'][key], d)}}} & {fmt(o['lecture'][key], d)} & {fmt(o['none'][key], d)}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def figure(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(-200_000, 300_000, 160)
    for mode, lab, col in (("none", "no shock", "0.6"), ("lecture", "lecture shock", "tab:blue"),
                           ("sheet", "sheet shock", "tab:red")):
        ax.hist(r["out"][mode]["l"], bins=bins, histtype="step", lw=1.3, color=col, label=lab, density=True)
    for v, lab in ((r["varK"], r"$\sqrt{10}$ VaR"), (r["reg"], r"$3\sqrt{10}$ VaR")):
        ax.axvline(v, color="k", ls="--", lw=0.8)
        ax.text(v, ax.get_ylim()[1] * 0.92, " " + lab, fontsize=8)
    ax.set_xlabel("10-day linearized loss ($)")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "losses.png"), dpi=200)


if __name__ == "__main__":
    r = run()
    c = checks(r)
    adj = run("stocks_adjclose.csv")
    seeds = [run(seed=s)["out"]["sheet"]["exc_var"] for s in range(1, 11)]
    print(f"n = {c['n']} returns {r['X'].index[0].date()}..{r['X'].index[-1].date()}")
    print("w", r["w"], "theta", r["theta"])
    print("mu", r["mu"], "sigma", r["s"], "rho", r["rho"])
    print(f"VaR1 {r['var1']:.2f} (MC {c['mc1']:.2f})  sqrt10 {r['varK']:.2f}  reg {r['reg']:.2f}")
    print("shock x", r["x"], "mu1c", r["mu1c"], "s1c", r["s1c"], "day-1 loss", r["day1_loss"])
    for mode, o in r["out"].items():
        print(f"{mode:8s} avg {o['avg']:10.0f} (se {o['se_avg']:.0f})  VaR {o['var']:10.0f}  "
              f"exc VaR {o['exc_var']:6.2f}%  exc reg {o['exc_reg']:6.3f}%  rho2 {np.median(o['rho2']):.3f}"
              f"  exact avg {o['exact_avg']:.0f}")
    print("iid-normal exact 10-day VaR", r["var_iid"], "mu2", r["mu2"])
    print("adj close: VaR1", adj["var1"], "rho", adj["rho"],
          {k: round(v, 3) for k, v in adj["out"]["sheet"].items() if k not in ("l", "rho2")})
    print("seed range of (c):", min(seeds), max(seeds))
    write_outputs(r, c, adj, seeds)
    figure(r)
