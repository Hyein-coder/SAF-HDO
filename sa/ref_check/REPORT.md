# SA literature-reference verification — REPORT

**Run:** 2026-08-03 · initial run · scope = SA-swept parameters only · citation authority = 202607 CEJ set.

**Headline: 3 match · 1 mismatch · 1 not-found · 6 source_unavailable · 5 not-verified (un-sourced by design).**

Environment caveat: this machine has **no PDF renderer on the Read tool's path** and most publishers (RSC, MDPI, Elsevier, Nature, Springer) **403/paywall automated fetch**. So most literature values could not be read directly. Disk-PDF sources were rescued via `pdftotext` (Bash) in the orchestrator. The paywalled LCA papers need institutional access or a human read — see Flags.

## Results — TEA

| Param             | Value used                  | Source · exact location                                             | Quote                                          | Verdict          | Note                                                                                                                          |
| ----------------- | --------------------------- | ------------------------------------------------------------------- | ---------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| feedstock_cost    | 0.06923 $/kg (2016$)        | INL Woody Feedstocks 2022 SOT (Sort_63795.pdf), p.16 · OSTI 2438337 | "delivered cost ... $69.23/dry ton (2016$)"    | **match**        | ×1000 = $69.23/ton exact. source_id '2016' = dollar-year, doc is 2022 SOT.                                                    |
| ng_price          | 0.0363 $/kg (76.57 ¢/MMBtu) | Jones et al. 2013 NREL/PNNL, **Table 14, ~p.23** (EIA 2011)         | "Natural Gas, $/1000 scf ... 5.1 ... EIA 2011" | **mismatch**     | Report = $5.1/1000 scf ≈ $4.95/MMBtu ≈ 0.245 $/kg; code ≈ **6.7× lower**. "76.57" not in report; cited p.71 wrong. Reconcile. |
| catalyst_lifetime | 3 yr                        | — (no cite)                                                         | —                                              | **not_verified** | Un-sourced SA range by design.                                                                                                |
| discount_rate     | 0.15                        | — (no cite)                                                         | —                                              | **not_verified** | Un-sourced SA range by design.                                                                                                |
| plant_life        | 15 yr                       | — (no cite)                                                         | —                                              | **not_verified** | Un-sourced SA range by design.                                                                                                |
| capex_mult        | 1.0 (±30%)                  | — (no cite)                                                         | —                                              | **not_verified** | Engineering uncertainty, not a lit value.                                                                                     |
| h2_price          | 1.3 $/kg                    | —                                                                   | —                                              | **not_verified** | Discarded from SA (internal H₂).                                                                                              |

## Results — LCA

| Param | Value used | Source · exact location | Quote | Verdict | Note |
|---|---|---|---|---|---|
| gwp_ng | 0.2221 kgCO₂e/kg | Mukherjee et al., Comm. Earth Environ. 6:16 (**2025**), doi:10.1038/s43247-024-01988-2, Abstract | "...Sabine Pass ... 0.220 ... tonnes of CO2e/tonne gas liquefied" | **match** | 0.220 t/t = 0.220 kg/kg ≈ 0.2221. See flag: code's "0.021 t/t" basis is the *regasification* figure (misattribution). |
| gwp_ng (md cite) | 0.2221 kgCO₂e/kg | Almena 2024, doi:10.1016/j.enconman.2024.118186 | — | **source_unavailable** | Elsevier paywall. (Code cite commsearth already matched.) |
| biomass_gwp | 0.1772 kgCO₂e/kg | Fitriasari 2023 · D'Ascenzo 2024 · Arfan 2023 (all DOIs) | — | **source_unavailable** ×3 | 403/paywall. D'Ascenzo covers lipid/HEFA feedstocks — may not contain a pyrolysis biomass GWP at all. |
| steam_ef | 0.11859247 kg/MJ | Fitriasari 2023, doi:10.1039/d3se00468f (LCI/SI) | — | **source_unavailable** | RSC 403. |
| cooling_ef | 7.048e-05 kg/kg | Fitriasari 2023, doi:10.1039/d3se00468f (LCI/SI) | — | **source_unavailable** | RSC 403. |
| wwt_ef | 0.431 kg/kg | Fitriasari 2023, doi:10.1039/d3se00468f (LCI/SI) | — | **source_unavailable** | RSC 403. **Dominant LCA contributor (~43%)** — prioritize manual check. |
| electricity_grid_factor | 0.67420026 kgCO₂e/kWh | Rojas-Michaga 2023 (117427) + npj 2026 (no DOI) | — | **source_unavailable** ×2 | OA PDF undecodable via fetch; npj has no locator. |
| allocation_method | mass vs energy | Osman et al. 2024, Environ. Chem. Lett., doi:10.1007/s10311-023-01694-z, p.8 / Fig.2 p.4 | "...mass-based or value-based allocation method is employed" | **match** | Supports mass allocation as a recognized choice. Caveat: names mass-vs-**value**, not mass-vs-energy. |
| allocation_method | mass vs energy | Batten et al. 2024, Sust. Energy Fuels 8:1924, doi:10.1039/d3se01470c, §2.2 p.1927, Table 3 p.1929 | "...offset emissions using system expansion credits." | **not_found** | Paper uses **system expansion/substitution**, NOT mass/energy allocation. Does not substantiate the framing. |

## Flags (most important first)

1. **MISMATCH — ng_price ~6.7× off its cited source.** NREL Jones 2013 Table 14 gives NG = $5.1/1000 scf (EIA 2011) ≈ 0.245 $/kg; code uses 0.0363 $/kg. The string "76.57 ¢/MMBtu" is not in the report and the cited "p.71" is wrong (value is Table 14, ~p.23). Either a code unit-error or a wrong citation. **Reconcile before the reviewer response.**

2. **NOT-FOUND — allocation citation (Batten 2024) doesn't support the claim.** Batten uses system expansion/substitution credits, not mass/energy co-product allocation. Osman 2024 supports mass-vs-*value* allocation, not mass-vs-*energy*. The SA's "mass vs energy allocation" framing is only partially backed by its cited sources.

3. **Code misattribution — gwp_ng basis.** Value 0.2221 kgCO₂e/kg is correct (matches Mukherjee 2025 liquefaction 0.220 t/t), but the code comment derives it from "0.021 t/t", which is that paper's *regasification* figure (a different stage, ~10× smaller). Right answer, wrong stated basis. Also: paper is **2025**, cited as 2024.

4. **code_vs_md_only — 6 LCA EFs cited only in the design-md, not in code.** biomass_gwp, steam/cooling/wwt EFs, electricity_grid_factor, and Almena's NG cite have no citation at the point the value is set (`LCA_PARAMETER.m` / `LCA_analysis_SAF.m`); the citations live only in `LCA_sensitivity_variables.md`. A code-only reviewer scan would find them un-cited.

5. **in_cej_refs = false for every LCA SA source + the INL feedstock report.** Only NREL Jones 2013 (CEJ ref [37]) is in the CEJ reference list. Notably, **CEJ attributes biomass GWP 0.1772 to GREET 2024**, whereas the SA md cites D'Ascenzo/Fitriasari/Arfan — a source disagreement to resolve.

6. **6 source_unavailable rows need a human/institutional read** (RSC, MDPI, Elsevier, Springer paywalls): all Fitriasari EFs (esp. **wwt_ef, the dominant contributor**), both biomass_gwp DOIs, Rojas-Michaga grid factor, Almena NG. These are not refuted — just unread here.

## How to close the gaps (re-run)

- **Disk PDFs are readable via `pdftotext`** (poppler + pymupdf are installed for Bash, just not on the Read tool's path). The verifier agent has been patched to use Bash `pdftotext` for `disk_pdf` sources — a re-run will read on-disk papers directly instead of returning source_unavailable.
- **Paywalled DOIs:** drop the publisher PDFs into a folder and re-map, or have someone with RSC/Elsevier/Springer access confirm the 6 flagged values. Re-run is incremental — only the still-`source_unavailable`/`mismatch` rows re-verify.

## Reference studies that report their own SA results

Text-keyword scan of the 260304 reference PDFs (`SAF references_manuscript\` + `SAF references_SI\`) for "sensitivity analysis", tornado, Monte Carlo, one-at-a-time. Deduped across folders (5 papers appear in both). Counts are a proxy, not a read — high-confidence rows are safe; borderline rows worth opening to confirm.

**High confidence — own SA results:**

| # (folder) | Study | Type | Signal |
|---|---|---|---|
| m5 | DMCO jet fuel from biomass — LCA + cost (Batten) | TEA+LCA | sens=9 |
| m10 | SAF from forest residues — TE + env | TEA+LCA | sens=4, mc=1 |
| m23 / SI12 | Lignocellulose → jet-range cycloalkanes | TEA | sens=8, tor=2 |
| m27 / SI2 | NREL fast pyrolysis + hydrotreating bio-oil (ng_price source) | TEA | sens=4 |
| m30 | CO₂ capture, phase-change solvent | TEA | mc=14, tor=1 |
| m33 | SAF via catalytic deox + hydrothermal gasification | TEA+LCA | sens=7, mc=6 |
| SI1 | BECCS — FT / bioSNG / OMEx | TEA+LCA | sens=6 |
| SI4 | Ethanol-to-jet blendstock upgrading | TEA | sens=15 |
| SI5 | Co-processing SAF at refineries | TEA+LCA | sens=3, mc=5 |
| SI8 | Hydroprocessed renewable jet — TE + resource | TEA | sens=5 |
| SI11 | Torrefaction + fast pyrolysis of pine | TEA | sens=3 |
| SI21 | US LNG GHG — process-model LCA (Mukherjee, gwp_ng source) | LCA | sens=7, tor=3 |

**Borderline** (low count, no tornado/MC — likely a short sensitivity section, confirm by opening):
- m21 / SI7 — pyrolysis vs Gasification-FT, waste tires (sens=3)
- m26 — gasoline/diesel via fast pyrolysis, design case (sens=2)

**Excluded** (keyword hit but no primary SA results):
- m28 Product & Process Design Principles — textbook (sens=28, oat=3: teaches the method)
- m34 / m35 — ISO 14040 / 14044 LCA standards (cite "sensitivity analysis" as a requirement)
- m1, m3, m6, m9 — reviews / landscape papers (passing mentions)
