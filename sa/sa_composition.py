"""Component-composition bars: how the TEA (MSP) cost buckets and LCA (GWP)
component fractions shift when each SA variable is pushed to its range low/high.
Style analogue of ../drHa_analysis.py's drHa_TEA_summary_3.png, but bars = SA
perturbations (baseline + LO/HI per variable) instead of process scenarios.

TEA base = case k (proposed best config C1, MSP ~0.95). LCA base = case j: the
SA Excel holds only the case_j_HOadjustment3 LCI row, so GWP stays at the 32.32
gCO2/MJ baseline. Run in the saf_hdo_shap env (imports shap_sa for the LCA model)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
import sa_model_ijk as M
import shap_sa as S                      # LCA component model + ranges

CASE = "k"   # proposed best config (C1), MSP ~0.95; LCA stays j (only LCI row available)
d = M.cases[CASE]
A = d["mSAF"] * M.TOP                     # annual SAF, kg/yr
GAS_CREDIT = d["mGAS"] * M.TOP * M.P_GAS  # constant across all TEA bars

SA_RC = {
    "figure.dpi": 150, "figure.facecolor": "white", "figure.autolayout": True,
    "axes.facecolor": "white", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.spines.top": True, "axes.spines.right": True, "axes.labelsize": 12,
    "axes.labelcolor": "black", "axes.labelweight": "500", "axes.grid": False,
    "xtick.labelsize": 10, "xtick.color": "black", "ytick.labelsize": 10,
    "ytick.color": "black", "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue LT Pro", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "font.weight": "300", "text.color": "black",
    "patch.edgecolor": "black", "patch.linewidth": 0.7, "patch.force_edgecolor": True,
    "legend.fontsize": 8, "legend.title_fontsize": 9, "legend.framealpha": 0.0,
    "legend.edgecolor": "#444444", "lines.linewidth": 0.7,
    "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
}
plt.rcParams.update(SA_RC)


def _bar_labels(base_label, groups):
    """Flatten (baseline, then LO/HI per variable) into a list of bar labels.
    groups = [(name, lo_lab, hi_lab), ...]."""
    labels = [base_label]
    for name, lo, hi in groups:
        labels += [f"{name}\n{lo}", f"{name}\n{hi}"]
    return labels


def stacked(ax, order, colors, values, credit_key=None):
    """values: list of dicts (component -> per-unit contribution). Stack the
    positive components; if credit_key is set, draw it as a hatched band at the
    top spanning [net, gross] and annotate the net total (= gross - credit)."""
    pos = [c for c in order if c != credit_key]
    x = np.arange(len(values))
    for i, v in enumerate(values):
        bottom = 0.0
        for comp in pos:
            ax.bar(x[i], v[comp], 0.7, bottom=bottom, color=colors[comp],
                   edgecolor="#444444", linewidth=0.7)
            bottom += v[comp]                       # bottom -> gross
        net = bottom
        if credit_key is not None:
            net = bottom - v[credit_key]
            ax.bar(x[i], v[credit_key], 0.7, bottom=net, color=colors[credit_key],
                   edgecolor="#444444", linewidth=0.7, hatch="//")
            ax.hlines(net, x[i]-0.35, x[i]+0.35, color="#444444", lw=1.0, zorder=5)
        ax.text(x[i], bottom, f"{net:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="500", color="#444444")
    return x


# ---------------- TEA: MSP cost-bucket composition ($/kg SAF) ----------------
TEA_ORDER = ["Feedstock", "NG", "Utility", "Labor", "Maintenance", "Overhead",
             "General Expenses", "Catalyst", "Capital & financing", "Gasoline credit"]
TEA_COLORS = {"Feedstock": "#FDD280", "NG": "#F5C270", "Utility": "#E4957D",
              "Labor": "#F08080", "Maintenance": "#B56565", "Overhead": "#D4A870",
              "General Expenses": "#C09763", "Catalyst": "#FADBC7",
              "Capital & financing": "#A1C1EC", "Gasoline credit": "#9FDAD3"}
TEA_CREDIT = "Gasoline credit"   # drawn downward (negative); net stack = MSP
_BIO_TON = M.BIO_PRICE0 * 1000
# (group name, npv kwarg, lo value, hi value, lo label, hi label)
TEA_LEVERS = [
    ("Feedstock", "bio_mult", 20/_BIO_TON, 120/_BIO_TON, "$20/t", "$120/t"),
    ("NG price",  "ng_mult",  0.50,        1.50,         "-50%",  "+50%"),
    ("Discount",  "irate",    0.00,        0.30,         "i=0%",  "i=30%"),
    ("Plant life","plife",    10.5,        19.5,         "10 yr", "20 yr"),
    ("Catalyst",  "cat_year", 1,           5,            "1 yr",  "5 yr"),
    ("Contingency","cont_frac",0.10,       0.40,         "10%",   "40%"),
]


def tea_composition(**kw):
    msp = M.solve_msp(d, cat_replace=True, **kw)
    cost_kw = {k: v for k, v in kw.items() if k not in ("irate", "plife")}
    c = M._costs(msp, d, cat_replace=True, **cost_kw)
    cash = {"Feedstock": c["C_feed"][0], "NG": c["C_feed"][1],
            "Utility": c["C_util"].sum(), "Labor": c["C_O"].sum(),
            "Maintenance": c["C_M"].sum(), "Overhead": c["C_OH"].sum() + c["C_prop"],
            "General Expenses": c["GE"], "Catalyst": c["C_cat_annual"]}
    cash_total = sum(cash.values())
    # levelized identity: revenue (SAF + gasoline credit) = cash OPEX + capital charge
    capital = msp * A + GAS_CREDIT - cash_total   # true capital & financing charge
    assert capital > 0, f"capital residual <=0 ({capital:.3g}) for {kw}"
    out = {k: v / A for k, v in cash.items()}
    out["Capital & financing"] = capital / A
    out["Gasoline credit"] = GAS_CREDIT / A       # subtracted in draw() -> net = MSP
    net = sum(out[c] for c in TEA_ORDER if c != TEA_CREDIT) - out[TEA_CREDIT]
    assert abs(net - msp) < 1e-6, "TEA net stack != MSP"
    return out


def build_tea():
    vals = [tea_composition()]
    for _, kwarg, lo, hi, *_ in TEA_LEVERS:
        vals.append(tea_composition(**{kwarg: lo}))
        vals.append(tea_composition(**{kwarg: hi}))
    labels = _bar_labels("Baseline", [(n, lo, hi) for n, _, _, _, lo, hi in TEA_LEVERS])
    return vals, labels


# ---------------- LCA: GWP component composition (gCO2/MJ) ----------------
LCA_ORDER = list(S.GWP_NAMES) + ["Fixed (emissions+steam+cooling)"]
LCA_COLORS = {"Biomass feedstock": "#B3CDE0", "Electricity": "#CCEBC5",
              "NG": "#DECBE4", "WWT": "#FED9A6",
              "Fixed (emissions+steam+cooling)": "#D9D9D9"}


def lca_composition(x):
    x = np.asarray(x, float)
    comps = {name: S._K * S._C[i] * x[i] * 1000.0 for i, name in enumerate(S.GWP_NAMES)}
    comps["Fixed (emissions+steam+cooling)"] = S._K * S._emi * 1000.0
    assert abs(sum(comps.values()) - float(S.f_gwp(x)[0])) < 1e-6, "LCA stack != f_gwp"
    return comps


def build_lca():
    base = np.ones(len(S.GWP_NAMES))
    vals = [lca_composition(base)]
    groups = []
    for i, name in enumerate(S.GWP_NAMES):
        lo, hi = S._GWP_RANGE[name]
        for b in (lo, hi):
            x = base.copy(); x[i] = b
            vals.append(lca_composition(x))
        groups.append((name.replace(" feedstock", ""),
                       f"{lo-1:+.0%}", f"{hi-1:+.0%}"))
    labels = _bar_labels("Baseline", groups)
    return vals, labels


def draw(vals, labels, order, colors, ylabel, fname, credit_key=None):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = stacked(ax, order, colors, vals, credit_key=credit_key)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.set_xlim(-0.7, len(vals) - 0.3)
    patches = [mpatches.Patch(color=colors[c], label=c) for c in order]
    leg = ax.legend(handles=patches, title="Components",
                    bbox_to_anchor=(1.01, 1), loc="upper left",
                    title_fontproperties={"weight": "500"})
    for p in leg.get_patches():
        p.set_edgecolor("#444444"); p.set_linewidth(0.7)
    out = os.path.join(HERE, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", out)


if __name__ == "__main__":
    tv, tl = build_tea()
    draw(tv, tl, TEA_ORDER, TEA_COLORS,
         r"MSP (\$ kg$^{-1}$ SAF)", "sa_composition_TEA.png", credit_key=TEA_CREDIT)
    print(f"  (TEA case {CASE}; hatched band = fixed gasoline co-product credit "
          f"{GAS_CREDIT/A:.3f} $/kg; tick line + label = net MSP)")
    lv, ll = build_lca()
    draw(lv, ll, LCA_ORDER, LCA_COLORS,
         r"GHG (g CO$_2$e MJ$^{-1}$)", "sa_composition_LCA.png")
