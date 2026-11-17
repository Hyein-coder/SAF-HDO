"""Decision trees: catalyst composition / preparation / measured properties -> Weibull3 (k, theta, lam).

n = 10 cases (a-j), so the trees are descriptive; LOOCV + permutation test say whether splits beat chance.
Run:  python cattree/tree.py [--selftest]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, permutation_test_score
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

HERE = Path(__file__).parent
OUT = HERE / "results"
MASS = {"La": 138.905, "Ti": 47.867, "Ce": 140.116, "Ni": 58.693, "O": 15.999}
# support oxides La2O3, TiO2, CeO2; A5's La2Ti0.5Ce0.5Ox support is taken as O5 (CeO2), not the
# Ce2O3-like ~O4.75 -- shifts A5 fractions by ~0.002, no split changes
O_PER = {"La": 1.5, "Ti": 2.0, "Ce": 2.0}
CAT = ["La", "Ti", "Ce", "Ni"]
FLAGS = ["prep_PD", "Ni_supported", "has_RuWZr"]
MEASURED = ["H2_uptake", "red_pct", "Ni_disp", "BET", "O_def"]
TARGETS = ["k", "theta", "lam"]
TREE = dict(max_depth=2, min_samples_leaf=2, random_state=0)


def fractions(raw):
    """Cation mole fractions; supported rows (Ni_wt set) convert wt% Ni + support formula to moles."""
    n = raw[[f"n_{e}" for e in CAT]].to_numpy(float)
    wt = raw["Ni_wt"].to_numpy(float)
    sup = ~np.isnan(wt)
    m_fu = sum(n[:, i] * (MASS[e] + O_PER[e] * MASS["O"]) for i, e in enumerate(CAT[:3]))
    n[sup, :3] *= ((100 - wt[sup]) / m_fu[sup])[:, None]
    n[sup, 3] = wt[sup] / MASS["Ni"]
    return pd.DataFrame(n / n.sum(1, keepdims=True), index=raw.index, columns=[f"x_{e}" for e in CAT])


def load():
    raw = pd.read_csv(HERE / "catalyst_features.csv", comment="#", index_col="case")
    X = pd.concat([fractions(raw), raw[FLAGS + MEASURED]], axis=1)
    y = pd.read_csv(HERE.parent / "distfit" / "results" / "weibull_laxmi.csv", index_col="case").loc[X.index, TARGETS]
    return raw, X, y


def main():
    OUT.mkdir(exist_ok=True)
    raw, X, y = load()
    X.round(4).join(raw["entry"]).to_csv(OUT / "features_used.csv")
    # "full" uses measured columns with NaN (trees route missing rows to whichever side fits best);
    # "complete" drops them, as a check that splits are not missing-data artifacts.
    # "_noC1": C1 copies A2's features (near-twin in LOO), so rerun without it.
    for tag, Xs in [("full", X), ("complete", X.dropna(axis=1))]:
        run(tag, Xs, y, raw)
        run(f"{tag}_noC1", Xs.drop(index="a"), y.drop(index="a"), raw)


def run(tag, X, y, raw):
    z = (y - y.mean()) / y.std()
    # multi-output tree on z-scored targets (equal weight); single trees on raw units for readable thresholds
    models = [("all_z", z.to_numpy())] + [(t, y[t].to_numpy()) for t in TARGETS]
    rows, imps, rules = [], {}, []
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    for (name, yy), ax in zip(models, axs.flat):
        m = DecisionTreeRegressor(**TREE).fit(X, yy)
        pred = cross_val_predict(DecisionTreeRegressor(**TREE), X, yy, cv=LeaveOneOut())
        _, _, p = permutation_test_score(DecisionTreeRegressor(**TREE), X, yy, cv=LeaveOneOut(),
                                         scoring="neg_mean_squared_error", n_permutations=1000, random_state=0)
        rows.append(dict(model=name, train_r2=r2_score(yy, m.predict(X)), loo_r2=r2_score(yy, pred),
                         loo_rmse=mean_squared_error(yy, pred) ** 0.5, perm_p=p))
        imps[name] = m.feature_importances_
        leaves = pd.Series(m.apply(X), index=X.index)
        members = "\n".join(f"  leaf {lf}: " + ", ".join(f"{c}({raw.at[c, 'entry']})" for c in g.index)
                            for lf, g in leaves.groupby(leaves))
        rules.append(f"== {name}\n{export_text(m, feature_names=list(X.columns), decimals=3)}{members}\n")
        plot_tree(m, feature_names=list(X.columns), filled=True, rounded=True, impurity=False,
                  precision=2, fontsize=9, ax=ax)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(OUT / f"trees_{tag}.png", dpi=200)
    cv = pd.DataFrame(rows).round(4)
    cv.to_csv(OUT / f"cv_{tag}.csv", index=False)
    pd.DataFrame(imps, index=X.columns).round(4).to_csv(OUT / f"importances_{tag}.csv")
    (OUT / f"rules_{tag}.txt").write_text("\n".join(rules), encoding="utf-8")
    print(f"########## {tag}: {list(X.columns)}\n")
    print("\n".join(rules))
    print(cv.to_string(index=False), "\n")


def selftest():
    rng = np.random.default_rng(0)
    Xt = pd.DataFrame({"a": rng.random(20), "b": rng.random(20)})
    Xt.loc[::4, "b"] = np.nan
    yt = np.where(Xt["a"] > 0.5, 3.0, 0.0)
    m = DecisionTreeRegressor(**TREE).fit(Xt, yt)
    assert m.tree_.feature[0] == 0 and np.allclose(m.predict(Xt), yt)       # planted split recovered
    raw, X, y = load()
    x = X.filter(like="x_")
    assert np.allclose(x.sum(axis=1), 1)
    assert np.allclose(x.loc["a"], x.loc["c"]) and np.allclose(x.loc["j"], x.loc["c"])  # C1, B4 = A2 composition
    assert np.isclose(x.at["h", "x_Ni"], (13.5 / 58.693) / (13.5 / 58.693 + 86.5 / (47.867 + 2 * 15.999)))
    assert list(X.index) == list("abcdefghij") and not y.isna().any().any()
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
