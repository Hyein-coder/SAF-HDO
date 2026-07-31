"""OAT LCA-GWP sensitivity from existing SA Excel (no Aspen re-run).

The SA workbook's LCA sheet already stores physical x emission-factor
contributions (Scenario_* columns), so perturbing an emission factor by (1+-p)
just scales that contribution group. Metric = allocated SAF GWP intensity
(Alloc_SAF_1, mass-basis) = S0 * const, so relative dGWP = relative dS.

5 in-use variables: feedstock GWP, electricity grid factor, natural-gas GWP,
utility EFs (steam+cool+WWT), co-product allocation method (mass vs energy).
"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

XLSX = r"D:\saf_hdo\TEA_LCA_non_CC_260731_catOPEX\data\SAF_Sensitivity_Analysis_260403_non_CC.xlsx"
CASE = None            # None -> first case (manuscript baseline: case_j_HOadjustment3_non_CC = 32.32 gCO2/MJ)
PCT  = dict(biomass=0.30, elec=0.20, ng=0.20, steam=0.20, cool=0.20, wwt=0.20)   # +- fractions
LHV_SAF, LHV_GAS = 48.168, 42.0

lca = pd.read_excel(XLSX, sheet_name="LCA")
tea = pd.read_excel(XLSX, sheet_name="TEA")

case = CASE or lca["Case"].iloc[0]
row  = lca[lca["Case"] == case]
assert len(row) == 1, f"case {case} not found"
row = row.iloc[0]
col = case.replace(".bkp", "")

def grp(prefix):
    cs = [c for c in lca.columns if c.startswith(prefix)]
    return float(pd.to_numeric(row[cs], errors="coerce").sum())

# emission contributions [kgCO2eq/hr]
contrib = {
    "Feedstock GWP (biomass)":  grp("Scenario_Feed_BM"),
    "Electricity grid factor":  grp("Scenario_elec_GRID"),
    "Natural gas GWP":          grp("Scenario_Feed_NG"),
    "Steam EF":                 grp("Scenario_steam"),
    "Cooling EF":               grp("Scenario_ref"),
    "WWT EF":                   grp("Scenario_WWT"),
}
emi = grp("Scenario_emi")
S0  = sum(contrib.values()) + emi
I0  = float(row["Alloc_SAF_1"])          # mass-basis allocated SAF GWP
k   = I0 / S0                            # S -> allocated-intensity constant

# productions for allocation-method swing
def tea_val(label):
    r = tea[tea.iloc[:, 0] == label]
    return float(r[col].iloc[0])
msaf = tea_val("SAF production [kg/hr]")
mgas = tea_val("Gasoline production [kg/hr]")
I_energy = S0 / (msaf * LHV_SAF + mgas * LHV_GAS)   # energy-basis alt

pmap = {"Feedstock GWP (biomass)": PCT["biomass"], "Electricity grid factor": PCT["elec"],
        "Natural gas GWP": PCT["ng"], "Steam EF": PCT["steam"],
        "Cooling EF": PCT["cool"], "WWT EF": PCT["wwt"]}

rows = []
for name, c in contrib.items():
    p = pmap[name]
    d = k * c * p                                   # symmetric swing magnitude
    rows.append(dict(Variable=name, Perturb=f"+-{p*100:.0f}%",
                     GWP_low=I0 - d, GWP_high=I0 + d,
                     Delta_low=-d, Delta_high=d, Swing_pct=200 * d / I0))
# allocation: discrete energy vs mass
d_alloc = I_energy - I0
rows.append(dict(Variable="Allocation (energy vs mass)", Perturb="energy/mass",
                 GWP_low=min(I0, I_energy), GWP_high=max(I0, I_energy),
                 Delta_low=min(0, d_alloc), Delta_high=max(0, d_alloc),
                 Swing_pct=100 * abs(d_alloc) / I0))

T = pd.DataFrame(rows).sort_values("Swing_pct", ascending=False).reset_index(drop=True)
print(f"Base case: {case}   baseline SAF GWP (mass-alloc) = {I0:.6g}")
print(f"energy-alloc SAF GWP = {I_energy:.6g}   S0 = {S0:.6g} kgCO2eq/hr\n")
with pd.option_context("display.width", 160, "display.max_columns", None):
    print(T.to_string(index=False))
out_csv = "lca_oat_results.csv"; T.to_csv(out_csv, index=False)

# tornado
plt.figure(figsize=(7.2, 3.8)); y = range(len(T))
plt.barh(y, T["Delta_low"],  color="#4d8cd9", label="low")
plt.barh(y, T["Delta_high"], color="#d9665a", label="high")
plt.yticks(list(y), T["Variable"]); plt.gca().invert_yaxis()
plt.axvline(0, color="k", lw=0.8); plt.xlabel("dGWP from baseline"); plt.grid(axis="x", alpha=0.3)
plt.title(f"LCA OAT tornado  (baseline GWP = {I0*1000:.2f} gCO2/MJ)"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("lca_oat_tornado.png", dpi=150)
print(f"\nsaved: {out_csv}, lca_oat_tornado.png")
