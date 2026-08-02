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
import shap

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

    shap.summary_plot(sv, X, feature_names=names, show=False, plot_size=(7.2, 4.0))
    plt.title(title); plt.xlabel(f"SHAP value ({unit})"); plt.tight_layout()
    png = os.path.join(HERE, f"shap_{tag}.png"); plt.savefig(png, dpi=150); plt.close()
    print(f"saved: {csv}, {png}")
    return sv, X, base_val

# ---------------- MSP (case j, nonlinear DCF) ----------------
MSP_NAMES = ["bio_mult", "ng_mult", "capex_mult", "irate", "plife", "cat_year"]
MSP_LO    = [0.50, 0.50, 0.70, 0.08, 10, 1]
MSP_HI    = [1.50, 1.50, 1.30, 0.20, 20, 5]
MSP_BASE  = [1.00, 1.00, 1.00, 0.15, 15, 3]
def f_msp(X):
    d = M.cases["j"]
    return np.array([M.solve_msp(d, bio_mult=r[0], ng_mult=r[1], capex_mult=r[2],
                                 irate=r[3], plife=r[4], cat_year=r[5], cat_replace=True)
                     for r in np.atleast_2d(X)])

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
_contrib = {"Feedstock GWP (biomass)": _grp("Scenario_Feed_BM"),
            "Electricity grid factor": _grp("Scenario_elec_GRID"),
            "Natural gas GWP":         _grp("Scenario_Feed_NG"),
            "Steam EF":                _grp("Scenario_steam"),
            "Cooling EF":              _grp("Scenario_ref"),
            "WWT EF":                  _grp("Scenario_WWT")}
_emi = _grp("Scenario_emi")
_C   = np.array(list(_contrib.values()))
_K   = float(_row["Alloc_SAF_1"]) / (_C.sum() + _emi)     # S -> allocated intensity
def f_gwp(X):
    X = np.atleast_2d(X)
    return _K * (_emi + X @ _C) * 1000.0                  # -> gCO2/MJ

GWP_NAMES = list(_contrib)
GWP_PCT   = [0.30, 0.20, 0.20, 0.20, 0.20, 0.20]          # biomass +-30%, rest +-20%
GWP_LO    = [1 - p for p in GWP_PCT]
GWP_HI    = [1 + p for p in GWP_PCT]
GWP_BASE  = [1.0] * 6

if __name__ == "__main__":
    sv_m, _, _ = run_shap(f_msp, MSP_NAMES, MSP_LO, MSP_HI, MSP_BASE,
                          "msp_casej", "MSP SHAP (case j)", "$/kg SAF")
    sv_g, Xg, _ = run_shap(f_gwp, GWP_NAMES, GWP_LO, GWP_HI, GWP_BASE,
                           "lca", "LCA GWP SHAP", "gCO2/MJ")
    # linearity check: KernelSHAP must recover analytic contribution shares k*c*(x-1)
    analytic = _K * _C * (Xg - 1.0) * 1000.0
    lin_err = np.abs(sv_g - analytic).max()
    assert lin_err < 1e-4, f"LCA SHAP != analytic k*c*(x-1), max err {lin_err:.2e}"
    print(f"\nLCA linearity check OK (SHAP == analytic shares, max err {lin_err:.1e})")
