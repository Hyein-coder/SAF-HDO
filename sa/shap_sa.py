"""KernelSHAP global sensitivity on the TEA (MSP) and LCA (GWP) models -- no ML
surrogate. Both models are fast Aspen-free callables, so KernelSHAP queries them
directly. Global importance = mean(|SHAP|) over a uniform sample of the OAT ranges;
background = single baseline row, so SHAP values attribute (output - baseline).

MSP: nonlinear DCF (sa_model_ijk.solve_msp), case j.
GWP: linear EF scaling (replicated from ../TEA_LCA_non_CC_260731_catOPEX/lca_oat_from_excel.py),
     so SHAP == analytic contribution shares k*c*(x-1) (asserted as a self-check).
"""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap

# feature-value colormap matching drHa_TEA_summary_3.png: CAPEX blue -> OPEX salmon
SHAP_CMAP = LinearSegmentedColormap.from_list("tea_bl_sal", ["#8FB5E8", "#F08080"])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sa_model_ijk as M

N = 1000                       # samples explained (global-SA population)
RNG = np.random.default_rng(0)

def run_shap(f, names, lo, hi, baseline, tag, title, unit):
    lo, hi, baseline = map(np.asarray, (lo, hi, baseline))
    X = RNG.uniform(lo, hi, size=(N, len(names)))
    expl = shap.KernelExplainer(f, baseline.reshape(1, -1))
    sv = np.asarray(expl.shap_values(X, silent=True))          # (N, n_features)
    base_val = float(np.ravel(expl.expected_value)[0])
    # SHAP efficiency: sum of attributions + baseline == model output
    err = np.abs(sv.sum(1) + base_val - f(X)).max()
    assert err < 1e-6, f"{tag}: SHAP efficiency broken, max err {err:.2e}"

    rank = (pd.DataFrame({"Variable": names, "mean_abs_SHAP": np.abs(sv).mean(0),
                          "mean_SHAP": sv.mean(0)})
            .sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True))
    csv = os.path.join(HERE, f"shap_{tag}.csv"); rank.to_csv(csv, index=False)
    print(f"\n== {tag} ==  baseline = {base_val:.5g} {unit}")
    print(rank.to_string(index=False))

    shap.summary_plot(sv, X, feature_names=names, show=False, plot_size=(7.2, 4.0), cmap=SHAP_CMAP)
    plt.xlabel(f"SHAP value ({unit})"); plt.tight_layout()   # no title (PLOTTING_STYLE.md)
    png = os.path.join(HERE, f"shap_{tag}.png"); plt.savefig(png, dpi=150); plt.close()
    print(f"saved: {csv}, {png}")
    return sv, X, base_val

# ---------------- MSP (case j, nonlinear DCF) ----------------
_BIO_TON = M.BIO_PRICE0 * 1000            # base biomass price, $/ton (bio_mult=1.0)
MSP_NAMES = ["Feedstock price", "NG price", "Discount rate", "Plant life", "Catalyst lifetime", "Contingency rate"]
MSP_LO    = [20/_BIO_TON,  0.50, 0.00, 10.5, 1, 0.10]   # feedstock $20/ton, NG -50%, i 0%, life -30%
MSP_HI    = [120/_BIO_TON, 1.50, 0.30, 19.5, 5, 0.40]   # feedstock $120/ton, NG +50%, i 30%, life +30%
MSP_BASE  = [1.00, 1.00, 0.15, 15, 3, 0.15]
def make_f_msp(case):
    d = M.cases[case]
    def f(X):
        return np.array([M.solve_msp(d, bio_mult=r[0], ng_mult=r[1], irate=r[2],
                                     plife=r[3], cat_year=r[4], cont_frac=r[5],
                                     cat_replace=True)
                         for r in np.atleast_2d(X)])
    return f

# ---------------- GWP (linear EF scaling) ----------------
# ponytail: replicated from lca_oat_from_excel.py rather than refactor that working
# script into an importable module. If the Excel schema changes, update both.
XLSX = os.path.join(HERE, "..", "TEA_LCA_non_CC_260731_catOPEX", "data",
                     "SAF_Sensitivity_Analysis_260403_non_CC.xlsx")
_lca = pd.read_excel(XLSX, sheet_name="LCA")
_row = _lca[_lca["Case"] == _lca["Case"].iloc[0]].iloc[0]
def _grp(pfx):
    cs = [c for c in _lca.columns if c.startswith(pfx)]
    return float(pd.to_numeric(_row[cs], errors="coerce").sum())
_contrib = {"Biomass feedstock": _grp("Scenario_Feed_BM"),
            "Electricity":       _grp("Scenario_elec_GRID"),
            "NG":                _grp("Scenario_Feed_NG"),
            "WWT":               _grp("Scenario_WWT")}
# steam + cooling dropped as SA variables (negligible); their fixed emission stays
# in the un-perturbed baseline so total GWP is unchanged.
_emi = _grp("Scenario_emi") + _grp("Scenario_steam") + _grp("Scenario_ref")
_C   = np.array(list(_contrib.values()))
_K   = float(_row["Alloc_SAF_1"]) / (_C.sum() + _emi)     # S -> allocated intensity
def f_gwp(X):
    X = np.atleast_2d(X)
    return _K * (_emi + X @ _C) * 1000.0                  # -> gCO2/MJ

GWP_NAMES = list(_contrib)
GWP_PCT   = [0.30, 0.20, 0.20, 0.20]                      # biomass +-30%, rest +-20%
GWP_LO    = [1 - p for p in GWP_PCT]
GWP_HI    = [1 + p for p in GWP_PCT]
GWP_BASE  = [1.0] * len(GWP_NAMES)

if __name__ == "__main__":
    # MSP SHAP per base scenario i, j, k
    msp_imp = {}
    for c in ("i", "j", "k"):
        sv, _, _ = run_shap(make_f_msp(c), MSP_NAMES, MSP_LO, MSP_HI, MSP_BASE,
                            f"msp_case{c}", f"MSP SHAP (case {c})", "$/kg SAF")
        msp_imp[c] = np.abs(sv).mean(0)
    # cross-case comparison: mean|SHAP| per feature, grouped bars
    cmp = pd.DataFrame(msp_imp, index=MSP_NAMES)
    cmp = cmp.loc[cmp.mean(1).sort_values(ascending=False).index]
    cmp.to_csv(os.path.join(HERE, "shap_msp_compare.csv"))
    print("\n== MSP mean|SHAP| across cases ==")
    print(cmp.to_string())
    ax = cmp.plot.barh(figsize=(7.2, 4.0), width=0.8)
    ax.invert_yaxis(); ax.set_xlabel("mean|SHAP|  ($/kg SAF)")
    ax.legend(title="case"); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "shap_msp_compare.png"), dpi=150); plt.close()
    print("saved: shap_msp_compare.csv, shap_msp_compare.png")

    sv_g, Xg, _ = run_shap(f_gwp, GWP_NAMES, GWP_LO, GWP_HI, GWP_BASE,
                           "lca", "LCA GWP SHAP", "gCO2/MJ")
    # linearity check: KernelSHAP must recover analytic contribution shares k*c*(x-1)
    analytic = _K * _C * (Xg - 1.0) * 1000.0
    lin_err = np.abs(sv_g - analytic).max()
    assert lin_err < 1e-4, f"LCA SHAP != analytic k*c*(x-1), max err {lin_err:.2e}"
    print(f"\nLCA linearity check OK (SHAP == analytic shares, max err {lin_err:.1e})")
