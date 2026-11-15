"""Compare distribution models on the 11 HDO SimDist carbon-number results.

Data: HDO_exp.xlsx Sheet1 -- Carbon 6..38 x experiments a..k, integer wt% (mass basis).
Random variable = carbon number n, frequency = wt%/100. Every model is a pmf on
n = 6..38: continuous models are binned as F(n+0.5) - F(n-0.5) and renormalized
over [5.5, 38.5] (nothing below C6 reported -> left-truncated; upper end truncated).
Fit = minimize cross-entropy -sum w_n log p_n (multinomial MLE on wt% = min KL).
Models are ranked on fit accuracy only (RMSE, KL); no parameter penalty.

Run:      python distfit/fit_dist.py
Selftest: python distfit/fit_dist.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
from scipy.special import gammainc, gammaincc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N = np.arange(6, 39)                      # carbon numbers C6..C38
EDGES = np.arange(5.5, 39.5)              # bin edges 5.5..38.5
NOISE_FLOOR = 0.5 / np.sqrt(3)            # sd of uniform +-0.5 wt% rounding error
JET, HEAVY = (N >= 8) & (N <= 16), N >= 17
RNG = np.random.default_rng(0)
TINY = 1e-300


def binned(dist):
    p = np.diff(dist.cdf(EDGES))
    return p / p.sum() if p.sum() > 0 else None


def asf_mass(alpha):
    # Anderson-Schulz-Flory, mass basis: W_n = n (1-alpha)^2 alpha^(n-1) ~ n alpha^n (chain unit = carbon)
    w = N * alpha ** N
    return w / w.sum()


def gg_prentice(mu, sig, Q):
    # generalized gamma, Prentice form: Q=0 lognormal, Q=sig gamma, Q=1 Weibull (all loc=0).
    # CDF via z = (ln x - mu)/sig: Q>0 -> P(a, a e^{Qz}), Q<0 -> 1 - P(a, a e^{Qz}), a = 1/Q^2
    # ponytail: exact lognormal for |Q|<1e-5 (a = 1/Q^2 too large); ~1e-7 kink at the switch
    z = (np.log(EDGES) - mu) / sig
    if abs(Q) < 1e-5:
        F = st.norm.cdf(z)
    else:
        a = 1 / Q**2
        with np.errstate(over="ignore"):
            u = a * np.exp(Q * z)
        F = gammainc(a, u) if Q > 0 else gammaincc(a, u)
    p = np.diff(F)
    return p / p.sum() if p.sum() > 0 else None


def nbinom_shift(r, q, n0=6):
    p = st.nbinom.pmf(N - n0, r, q)
    return p / p.sum() if p.sum() > 0 else None


# name -> (pmf(*params), moment-based x0(mean, var), bounds, param names, shared idx for pooled fit)
# ponytail: Nelder-Mead + random restarts; switch to a global optimizer if restarts disagree a lot
def _ln(m, v):
    s2 = np.log1p(v / m**2)
    return np.sqrt(s2), m / np.sqrt(1 + v / m**2)


MODELS = {
    "normal":   (lambda mu, sd: binned(st.norm(mu, sd)),
                 lambda m, v: [m, np.sqrt(v)], [(0, 40), (0.3, 40)], ["mu", "sd"], [1]),
    "lognorm":  (lambda s, sc: binned(st.lognorm(s, scale=sc)),
                 lambda m, v: list(_ln(m, v)), [(0.01, 3), (1, 60)], ["s", "scale"], [0]),
    "lognorm3": (lambda s, sc, th: binned(st.lognorm(s, loc=th, scale=sc)),
                 lambda m, v: [*_ln(m - 3, v), 3.0], [(0.01, 5), (0.1, 80), (-40, 20)],
                 ["s", "scale", "theta"], [0, 2]),
    "gamma":    (lambda a, sc: binned(st.gamma(a, scale=sc)),
                 lambda m, v: [m**2 / v, v / m], [(0.1, 200), (0.01, 40)], ["a", "scale"], [0]),
    "weibull3": (lambda k, th, lam: binned(st.weibull_min(k, loc=th, scale=lam)),
                 lambda m, v: [1.5, 4.0, m - 4], [(0.3, 10), (-40, 20), (0.1, 80)],
                 ["k", "theta", "lambda"], [0, 1]),
    "asf":      (asf_mass, lambda m, v: [0.7], [(0.01, 0.999)], ["alpha"], None),
    # ASF mass basis counted from a seed at the first reported carbon: W ~ m alpha^m, m = n - n0 + 1
    # (= nbinom with r = 2, q = 1 - alpha); pmf bound per case in pmf_for
    "asf_shift": (None, lambda m, v: [0.8], [(0.01, 0.999)], ["alpha"], None),
    "nbinom":   (nbinom_shift, lambda m, v: [3.0, 3.0 / (3.0 + m - 6)],
                 [(0.05, 500), (0.001, 0.999)], ["r", "q"], [0]),
    "gengamma": (gg_prentice, lambda m, v: [np.log(_ln(m, v)[1]), _ln(m, v)[0], 0.5],
                 [(0, 5), (0.01, 3), (-5, 5)], ["mu", "sigma", "Q"], [1, 2]),
}
NESTED = [("gengamma", "gamma"), ("gengamma", "lognorm"), ("lognorm3", "lognorm"),
          ("nbinom", "asf_shift")]


def pmf_for(name, w):
    """Model pmf for one case. nbinom support starts at the first reported carbon (C7 in b-f),
    matching the theta cap of the 3p models; n0 is data-set, not counted in k."""
    n0 = N[w > 0].min()
    if name == "nbinom":
        return lambda r, q: nbinom_shift(r, q, n0)
    if name == "asf_shift":
        return lambda alpha: nbinom_shift(2, 1 - alpha, n0)
    return MODELS[name][0]


def case_bounds(name, w):
    """Static bounds, with location theta capped so the first reported carbon bin keeps mass."""
    cap = N[w > 0].min() + 0.5 - 1e-3
    return [(lo, min(hi, cap)) if nm == "theta" else (lo, hi)
            for (lo, hi), nm in zip(MODELS[name][2], MODELS[name][3])]


def load():
    df = pd.read_excel(ROOT / "HDO_exp.xlsx").fillna(0).set_index("Carbon")
    assert set(df.index) <= set(N), f"carbon numbers outside C6-C38: {sorted(set(df.index) - set(N))}"
    df = df.reindex(N, fill_value=0)
    return df / df.sum()                  # i sums to 99 in some copies; normalize per column


def xent(p, w):
    return np.inf if p is None or not np.all(np.isfinite(p)) else -np.sum(w * np.log(np.maximum(p, TINY)))


def fit(name, w, n_starts=5):
    pmf, x0f = pmf_for(name, w), MODELS[name][1]
    bounds = case_bounds(name, w)
    lo, hi = np.array(bounds, float).T
    m = np.sum(N * w); v = np.sum((N - m) ** 2 * w)
    starts = [np.clip(x0f(m, v), lo, hi)] + [RNG.uniform(lo, hi) for _ in range(n_starts - 1)]
    f = lambda x: xent(pmf(*x), w)
    best = None
    for x0 in starts:
        with np.errstate(all="ignore"):
            r = minimize(f, x0, method="Nelder-Mead", bounds=bounds,
                         options=dict(maxiter=20000, maxfev=20000, xatol=1e-8, fatol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    return best


def hessian_se(name, x, w, npseudo=100, h=1e-4):
    """SE from finite-difference Hessian of npseudo * cross-entropy, over params not at a bound.
    Returns (se, at_bound mask, positive-definite flag)."""
    pmf = pmf_for(name, w)
    lo, hi = np.array(case_bounds(name, w), float).T
    at_b = np.isclose(x, lo, rtol=1e-3, atol=1e-6) | np.isclose(x, hi, rtol=1e-3, atol=1e-6)
    f = lambda z: npseudo * xent(pmf(*z), w)
    k = len(x); H = np.empty((k, k)); d = h * np.maximum(np.abs(x), 1.0); I = np.eye(k)
    with np.errstate(all="ignore"):
        for i in range(k):
            for j in range(k):
                ei, ej = I[i] * d[i], I[j] * d[j]
                H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * d[i] * d[j])
    se = np.full(k, np.nan); idx = np.flatnonzero(~at_b)
    Hs = H[np.ix_(idx, idx)]
    pd_ok = bool(idx.size and np.all(np.isfinite(Hs)) and np.linalg.eigvalsh((Hs + Hs.T) / 2).min() > 0)
    if idx.size and pd_ok:
        se[idx] = np.sqrt(np.diag(np.linalg.inv(Hs)))
    return se, at_b, pd_ok


def metrics(name, x, w):
    p = pmf_for(name, w)(*x); k = len(x)
    H = xent(p, w); S = -np.sum(w[w > 0] * np.log(w[w > 0]))
    e = 100 * (p - w)
    row = dict(model=name, k=k, xent=H, kl=H - S,
               rmse=np.sqrt(np.mean(e[(N >= N[w > 0].min()) & (N <= N[w > 0].max())] ** 2)), maxabs=np.max(np.abs(e)),
               d_jet=100 * (p[JET].sum() - w[JET].sum()), d_heavy=100 * (p[HEAVY].sum() - w[HEAVY].sum()))
    return row, p


def pooled(name, W, indiv):
    """One shared set of shape params (MODELS[name][4]) across all experiments, rest per experiment.
    Start: try median and every case's own shared values, refit free params per case, keep best."""
    sh = MODELS[name][4]
    k = len(MODELS[name][2]); free = [i for i in range(k) if i not in sh]; E = W.shape[1]
    cb = [case_bounds(name, W[c].values) for c in W.columns]
    bounds = [(max(b[i][0] for b in cb), min(b[i][1] for b in cb)) for i in range(k)]  # tightest over cases
    X = np.array([indiv[(name, c)] for c in W.columns])
    zb = [bounds[i] for i in sh] + [bounds[i] for i in free] * E

    def unpack(z):
        xs = np.empty((E, k)); xs[:, sh] = z[:len(sh)]; xs[:, free] = z[len(sh):].reshape(E, len(free))
        return xs

    def refit_free(shared):
        xs = X.copy(); xs[:, sh] = shared; tot = 0.0
        for e, c in enumerate(W.columns):
            def g(fz, e=e):
                x = xs[e].copy(); x[free] = fz
                return xent(pmf_for(name, W[c].values)(*x), W[c].values)
            r = minimize(g, np.clip(xs[e, free], *np.array([bounds[i] for i in free]).T),
                         method="Nelder-Mead", bounds=[bounds[i] for i in free])
            xs[e, free] = r.x; tot += r.fun
        return tot, np.r_[shared, xs[:, free].ravel()]

    obj = lambda z: sum(xent(pmf_for(name, W[c].values)(*x), W[c].values) for x, c in zip(unpack(z), W.columns))
    lo_s, hi_s = np.array([bounds[i] for i in sh]).T
    cands = [np.clip(np.median(X[:, sh], axis=0), lo_s, hi_s)] + [np.clip(x, lo_s, hi_s) for x in X[:, sh]]
    with np.errstate(all="ignore"):
        _, z0 = min((refit_free(cs) for cs in cands), key=lambda t: t[0])
        r = minimize(obj, z0, method="L-BFGS-B", bounds=zb)
        r = minimize(obj, r.x, method="Nelder-Mead", bounds=zb, options=dict(maxiter=50000, maxfev=50000))
    return unpack(r.x), r.fun, len(sh) + E * len(free)


def plots(W, P, M, top3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # SA_RC from PLOTTING_STYLE.md (copied: sa/sa_composition.py runs the SA model on import)
    plt.rcParams.update({
        "figure.dpi": 150, "figure.facecolor": "white", "figure.autolayout": True,
        "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
        "axes.labelsize": 12, "axes.labelweight": "500", "axes.grid": False,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue LT Pro", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "font.weight": "300", "patch.linewidth": 0.7,
        "legend.fontsize": 8, "legend.framealpha": 0.0, "lines.linewidth": 0.7,
        "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
    })
    fig_dir = ROOT / "figures"; fig_dir.mkdir(exist_ok=True)
    cols = ["#B56565", "#4F81BD", "#E4957D"]  # drHa warm/cool, darkened for line contrast

    fig, axs = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
    for ax, c in zip(axs.flat, W.columns):
        ax.bar(N, 100 * W[c], 0.8, color="#FDD280", edgecolor="#444444", linewidth=0.5, label="SimDist")
        for m, col in zip(top3, cols):
            ax.plot(N, 100 * P[(m, c)], "-o", color=col, lw=1.2, ms=2.5, label=m)
        ax.set_title(f"case {c}")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    axs.flat[-1].axis("off")
    axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(), loc="center")
    for ax in axs[-1]: ax.set_xlabel("Carbon number")
    for ax in axs[:, 0]: ax.set_ylabel("wt%")
    fig.savefig(fig_dir / "distfit_grid.png"); fig.savefig(fig_dir / "distfit_grid.svg"); plt.close(fig)

    from matplotlib.colors import LinearSegmentedColormap
    piv = M.pivot(index="model", columns="case", values="rmse").loc[list(MODELS)]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(piv.values, cmap=LinearSegmentedColormap.from_list("bl_sal", ["#8FB5E8", "#F08080"]),
                   aspect="auto")
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel("Experiment"); fig.colorbar(im, label="RMSE (wt%)")
    fig.savefig(fig_dir / "distfit_rmse_heatmap.png"); fig.savefig(fig_dir / "distfit_rmse_heatmap.svg"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    order = M.groupby("model").rmse.mean().sort_values().index
    for i, m in enumerate(order):
        r = M[M.model == m].rmse
        ax.bar(i, r.mean(), 0.6, color="#A1C1EC", edgecolor="#444444")
        ax.plot(np.full(len(r), i) + RNG.uniform(-0.15, 0.15, len(r)), r, "o", color="#444444", ms=3)
    ax.axhline(NOISE_FLOOR, color="#B56565", ls="--", lw=1, label=f"rounding floor {NOISE_FLOOR:.2f} wt%")
    ax.set_xticks(range(len(order)), order, rotation=30); ax.set_ylabel("RMSE (wt%)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2); ax.legend()
    fig.savefig(fig_dir / "distfit_rmse.png"); fig.savefig(fig_dir / "distfit_rmse.svg"); plt.close(fig)


def laxmi(W, X):
    """Map per-case weibull3 (k, theta, lambda) onto the Laxminarasimhan et al. (1996) rate law.

    Their Eq. 4: k(th) = k_max * th^(1/alpha), th = normalized TBP (0 at lightest, 1 at heaviest species).
    Taking th = (n - n_l)/(n_h - n_l), first-order loss exp(-k(th) tau) of one species is a Weibull
    survival exp(-((n - theta)/lambda)^k) with k = 1/alpha, theta = n_l, lambda = (n_h - n_l)(k_max tau)^-alpha.
    Formal analogy only: equating it with the *cumulative* mass heavier than n also needs no formation
    term (Eq. 8 integral), a feed uniform in k, TBP linear in n -- see REPORT.md.
    kmax_tau_C38 = ((38.5 - theta)/lambda)^k, i.e. n_h = 38.5; it is -ln of the *untruncated* Weibull
    mass above C38 (the fitted, window-renormalized pmf has none). sf_below/sf_above = untruncated
    Weibull mass below C5.5 / above C38.5, to show how much the window truncation matters.
    """
    rows = []
    for c in W.columns:
        k, th, lam = X[("weibull3", c)]
        d = st.weibull_min(k, loc=th, scale=lam)
        rows.append(dict(case=c, k=k, theta=th, lam=lam, alpha=1 / k,
                         kmax_tau_C38=((EDGES[-1] - th) / lam) ** k,
                         sf_below=1 - d.sf(EDGES[0]), sf_above=d.sf(EDGES[-1])))
    return pd.DataFrame(rows)


def plot_laxmi(W, L):
    """-ln(fraction heavier than n) vs (n - theta), log-log. Line = window-truncated weibull3 (what was
    fitted); dashed = untruncated power law (n - theta)^(1/alpha) / lambda^(1/alpha)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(14, 9))
    for ax, (c, r) in zip(axs.flat, L.set_index("case").iterrows()):
        H = 1 - np.cumsum(W[c].values)                       # mass fraction heavier than n (at edge n+0.5)
        x = EDGES[1:] - r.theta; ok = (H > 1e-6) & (H < 1 - 1e-9) & (x > 0)   # drop empty leading bins
        ax.loglog(x[ok], -np.log(H[ok]), "o", color="#444444", ms=3, label="SimDist")
        n = np.linspace(EDGES[1:][ok].min(), EDGES[-1] - 0.05, 200)
        sf = st.weibull_min(r.k, loc=r.theta, scale=r.lam).sf
        Htr = (sf(n) - sf(EDGES[-1])) / (sf(EDGES[0]) - sf(EDGES[-1]))
        ax.loglog(n - r.theta, -np.log(Htr), "-", color="#B56565", lw=1.2, label="weibull3 (fitted, truncated)")
        ax.loglog(n - r.theta, ((n - r.theta) / r.lam) ** r.k, "--", color="#4F81BD", lw=1,
                  label=rf"power law, $1/\alpha$ = {r.k:.2f}")
        ax.set_title(f"case {c}"); ax.legend()
    axs.flat[-1].axis("off")
    for ax in axs[-1]: ax.set_xlabel(r"$n - \theta$ (carbon number)")
    for ax in axs[:, 0]: ax.set_ylabel("−ln(wt fraction heavier than n)")
    fig_dir = ROOT / "figures"
    fig.savefig(fig_dir / "distfit_laxmi.png"); fig.savefig(fig_dir / "distfit_laxmi.svg"); plt.close(fig)


def main():
    W = load()
    rows, prow, P, X = [], [], {}, {}
    for c in W.columns:
        w = W[c].values
        for m in MODELS:
            r = fit(m, w)
            if not np.isfinite(r.fun):
                raise RuntimeError(f"no finite fit: {m}/{c}")   # downstream tables assume every pair
            row, p = metrics(m, r.x, w)
            se, at_b, pd_ok = hessian_se(m, r.x, w)
            # converged = Nelder-Mead stopped before maxiter (weak); at_bound / not pd_ok are the real flags
            row.update(case=c, converged=bool(r.success),
                       at_bound=",".join(nm for nm, b in zip(MODELS[m][3], at_b) if b),
                       unidentifiable=not pd_ok,
                       # weak = SE > |value| for a scale/shape param (location-type theta/mu/Q excluded)
                       weak=",".join(nm for nm, v, e, b in zip(MODELS[m][3], r.x, se, at_b)
                                     if not b and nm not in ("theta", "mu", "Q") and not e <= abs(v)))
            rows.append(row); P[(m, c)] = p; X[(m, c)] = r.x
            prow += [dict(case=c, model=m, param=nm, value=v, se=s)
                     for nm, v, s in zip(MODELS[m][3], r.x, se)]
            if not r.success:
                print(f"WARN not converged: {m}/{c}: {r.message}")
    M = pd.DataFrame(rows)

    # nested models must fit at least as well as their children (else optimizer issue)
    for big, small in NESTED:
        for c in W.columns:
            hb, hs = (M[(M.model == x) & (M.case == c)].xent.item() for x in (big, small))
            if hb > hs + 1e-6:
                print(f"WARN nesting violated: {big} worse than {small} on {c} ({hb:.6f} > {hs:.6f})")

    rank = pd.DataFrame({
        "k": M.groupby("model").k.first(),
        # accuracy only, no parameter penalty
        "wins_rmse": M.loc[M.groupby("case").rmse.idxmin()].model.value_counts(),
        "wins_kl": M.loc[M.groupby("case").kl.idxmin()].model.value_counts(),
        "mean_rmse": M.groupby("model").rmse.mean(),
        "mean_kl": M.groupby("model").kl.mean(),
        "mean_maxabs": M.groupby("model").maxabs.mean(),
        "max_rmse": M.groupby("model").rmse.max(),
        "mean_abs_d_jet": M.groupby("model").d_jet.apply(lambda s: s.abs().mean()),
        "mean_abs_d_heavy": M.groupby("model").d_heavy.apply(lambda s: s.abs().mean()),
        "n_unidentifiable": M.groupby("model").unidentifiable.sum(),
        "n_at_bound": M.groupby("model").at_bound.apply(lambda s: (s.fillna("") != "").sum()),
        "n_weak": M.groupby("model").weak.apply(lambda s: (s.fillna("") != "").sum()),
    }).fillna({"wins_rmse": 0, "wins_kl": 0}).sort_values("mean_rmse")

    # pooled shared-shape fit for the top-3 models that define shared params
    top3 = list(rank.index[:3])
    pool_rows = []
    for m in [m for m in rank.index if MODELS[m][4] is not None][:3]:
        xs, H, kp = pooled(m, W, X)
        Hind = M[M.model == m].xent.sum(); kind = len(MODELS[m][3]) * W.shape[1]
        if H < Hind - 1e-6:
            print(f"WARN pooled {m} beats individual fits ({H:.5f} < {Hind:.5f}): individual fits suboptimal")
        rm = [metrics(m, x, W[c].values)[0]["rmse"] for x, c in zip(xs, W.columns)]
        pool_rows.append(dict(model=m, shared=",".join(MODELS[m][3][i] for i in MODELS[m][4]),
                              shared_value=",".join(f"{v:.4g}" for v in xs[0, MODELS[m][4]]),
                              k_pooled=kp, k_indiv=kind,
                              mean_kl_pooled=(H - Hind) / W.shape[1] + M[M.model == m].kl.mean(),
                              mean_kl_indiv=M[M.model == m].kl.mean(),
                              mean_rmse_pooled=np.mean(rm), mean_rmse_indiv=M[M.model == m].rmse.mean()))
    pool = pd.DataFrame(pool_rows)

    out = HERE / "results"; out.mkdir(exist_ok=True)
    M.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(prow).to_csv(out / "params.csv", index=False)
    rank.to_csv(out / "ranking.csv")
    pool.to_csv(out / "pooled.csv", index=False)
    pd.set_option("display.width", 200)
    print(rank.round(3).to_string()); print(); print(pool.round(4).to_string())
    L = laxmi(W, X); L.to_csv(out / "weibull_laxmi.csv", index=False)
    print(); print(L.round(3).to_string())
    plots(W, P, M, top3)
    plot_laxmi(W, L)


def selftest():
    # exact pmf -> params recovered; integer-rounded pmf -> fit within rounding noise
    for name, true in [("gamma", [6.0, 2.0]), ("weibull3", [1.8, 4.0, 8.0]), ("lognorm", [0.4, 11.0])]:
        w = MODELS[name][0](*true)
        r = fit(name, w)
        assert np.allclose(r.x, true, rtol=0.02), (name, r.x, true)
        wr = np.round(100 * w); wr = wr / wr.sum()
        row, _ = metrics(name, fit(name, wr).x, wr)
        assert row["rmse"] < 0.45, (name, row["rmse"])
    assert np.isclose(asf_mass(0.7).sum(), 1) and np.isclose(nbinom_shift(3, 0.3).sum(), 1)
    assert not np.allclose(asf_mass(0.7), nbinom_shift(2, 0.3))          # ASF is not NB(r=2)
    m = N - 6 + 1; ref = m * 0.7 ** m / np.sum(m * 0.7 ** m)                # seeded ASF = NB(r=2)
    assert np.allclose(pmf_for("asf_shift", np.ones_like(N))(0.7), ref)
    mu, sg = np.log(11.0), 0.4
    assert np.allclose(gg_prentice(mu, sg, 2e-3), binned(st.lognorm(sg, scale=11.0)), atol=1e-3)
    assert np.allclose(gg_prentice(mu, sg, sg), binned(st.gamma(1 / sg**2, scale=11.0 * sg**2)), atol=1e-9)
    assert np.allclose(gg_prentice(mu, sg, 1.0), binned(st.weibull_min(1 / sg, scale=11.0)), atol=1e-9)
    W = load()
    assert W.shape == (33, 11) and np.allclose(W.sum(), 1)
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
