# LCA Sensitivity Analysis — Variable Selection

Scope: **non-CC case only**, HDO of biomass fast-pyrolysis oil, internal H₂ production, experimental yields fixed.
Model files: `LCA_PARAMETER.m`, `LCA_analysis_SAF.m`.
Date: 2026-07-31.

## Variables IN USE

| #   | Variable (code)                                                                                                        | Rationale                                                | Range                                 | Citations                                                 |
| --- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| 1   | Feedstock GWP — `GWP_biomass` (0.1772), `LCA.BM`                                                                       | Biomass cultivation/collection often top GHG contributor | ±20–50%                               | D'Ascenzo 2024; Fitriasari 2023; Arfan 2023               |
| 3   | Electricity grid factor — `EL_GRID` (0.674) ✅ integrated into OAT (`LCA_OAT_SAF.m`), replaces old 5-scenario structure | Drives GWP; grid decarbonization lever                   | grid ↔ renewable (0.016–0.674 kg/kWh) | Rojas-Michaga 2023 (Monte Carlo); npj Sust. Mobility 2026 |
| 5   | Natural gas — `GWP_NG` (0.2221), `NG_kg`                                                                               | Process heat / feed                                      | ±20%                                  | Almena 2024                                               |
| 6   | Co-product allocation method — mass vs energy (`ratio_SAF`, /42, /48.168)                                              | Method choice shifts SAF burden materially               | energy ↔ mass ↔ system expansion      | Batten 2024; Osman 2024 (review)                          |
| 7   | Utility emission factors — `STEAM` (0.1186), `COOL` (7.05e-5), `WWT` (0.431)                                           | Secondary but cheap to test                              | ±20%                                  | Fitriasari 2023                                           |

## Variables ON HOLD

| # | Variable | Reason |
|---|---|---|
| 4 | Process/conversion yield (SAF+gasoline massflow, HDO conversion) | Keep fixed to **experimental data**; not perturbed. Field reports yield as largest GHG driver (Fitriasari 2023; Arfan 2023) — revisit only if experimental uncertainty is quantified. |

## Variables DISCARDED

| # | Variable | Reason |
|---|---|---|
| 2 | Hydrogen source — `H2_GREY` vs `H2_GREEN` | No external H₂; H₂ produced internally. Not applicable. |
| 8 | CO₂-capture credit — `O(1)` factor | **Non-CC case only.** Out of scope. |
| 9 | Biomass transport distance | **Not implemented** in current model (verified: no transport term in `LCA_*.m`). Out of scope unless added later. |
| 10 | GWP characterization / biogenic-C accounting, time horizon | Out of scope. |

## References (recent 3 yr)

- Fitriasari EI, Won W, Liu J. *Sustainability assessment of biojet fuel produced from pyrolysis oil of woody biomass.* Sustainable Energy & Fuels, 2023. doi:10.1039/d3se00468f — closest analog (pyrolysis oil → biojet, woody biomass).
- Rojas-Michaga MF, Michailos S, Cardozo E, et al. *SAF production through power-to-liquid (PtL): combined TEA and LCA.* Energy Conversion and Management, 2023. doi:10.1016/j.enconman.2023.117427 — Monte Carlo GWP, electricity-driven.
- Almena A, et al. *Reducing environmental impact of aviation via SAF with integrated CCS.* Energy Conversion and Management, 2024. doi:10.1016/j.enconman.2024.118186.
- D'Ascenzo F, et al. *Comparative LCA of SAF production from different biomasses.* Sustainability, 2024. doi:10.3390/su16166875.
- Batten R, et al. *A SAF pathway from biomass: life-cycle environmental and cost evaluation.* Sustainable Energy & Fuels, 2024. doi:10.1039/d3se01470c — allocation methods.
- Arfan M, et al. *LCA and LCC of hydrogen production from biowaste and biomass in Sweden.* Energy Conversion and Management, 2023. doi:10.1016/j.enconman.2023.117262 — feedstock-to-yield ratio + energy source sensitivity.
- Osman AI, et al. *LCA and TEA of sustainable bioenergy production: a review.* Environmental Chemistry Letters, 2024. doi:10.1007/s10311-023-01694-z.

## Method

1. **OAT tornado** — perturb each IN-USE variable ±10–20%, rank by ΔGWP.
2. **Monte Carlo** — distributions on background EFs (electricity, feedstock, NG, utilities), 10,000 runs, report 95% CI (per Rojas-Michaga 2023).

## Implementation & results

- Script: `lca_oat_from_excel.py` — reads EF-scaled contribution columns (`Scenario_*`) straight from the SA Excel and scales each group; **no Aspen/MATLAB re-run needed**. (`LCA_OAT_SAF.m` = MATLAB variant that re-runs Aspen, for pipeline integration.)
- Baseline data: `data/SAF_Sensitivity_Analysis_260403_non_CC.xlsx`, case `260401_pyrolysis_oil_CC_case_j_HOadjustment3_non_CC` = **manuscript baseline 32.32 gCO₂/MJ** (mass-basis allocation). Utility EFs split into steam / cooling / WWT.
- Perturbations: biomass ±30%, elec/NG/steam/cool/WWT ±20%, allocation = energy-vs-mass discrete.

OAT swing (base 32.32 gCO₂/MJ; contribution share of S₀ in parens):

| Variable | ±% | Swing | Note |
|---|---|---|---|
| WWT EF | ±20% | 17.3% | dominant (43% of S₀) |
| Feedstock GWP (biomass) | ±30% | 13.6% | (23%) |
| Natural gas GWP | ±20% | 7.4% | |
| Electricity grid factor | ±20% | 6.3% | (16%) |
| Allocation (energy vs mass) | — | 1.6% | energy = 32.85 g/MJ |
| Steam EF | ±20% | 0.004% | negligible |
| Cooling EF | ±20% | ~0% | negligible |

Outputs: `lca_oat_results.csv`, `lca_oat_tornado.png`.
