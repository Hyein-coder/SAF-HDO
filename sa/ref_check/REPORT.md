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

| # (folder) | Study                                                         | Type    | Signal        |
| ---------- | ------------------------------------------------------------- | ------- | ------------- |
| m5         | DMCO jet fuel from biomass — LCA + cost (Batten)              | TEA+LCA | sens=9        |
| m10        | SAF from forest residues — TE + env                           | TEA+LCA | sens=4, mc=1  |
| m23 / SI12 | Lignocellulose → jet-range cycloalkanes                       | TEA     | sens=8, tor=2 |
| m27 / SI2  | NREL fast pyrolysis + hydrotreating bio-oil (ng_price source) | TEA     | sens=4        |
| m30        | CO₂ capture, phase-change solvent                             | TEA     | mc=14, tor=1  |
| m33        | SAF via catalytic deox + hydrothermal gasification            | TEA+LCA | sens=7, mc=6  |
| SI1        | BECCS — FT / bioSNG / OMEx                                    | TEA+LCA | sens=6        |
| SI4        | Ethanol-to-jet blendstock upgrading                           | TEA     | sens=15       |
| SI5        | Co-processing SAF at refineries                               | TEA+LCA | sens=3, mc=5  |
| SI8        | Hydroprocessed renewable jet — TE + resource                  | TEA     | sens=5        |
| SI11       | Torrefaction + fast pyrolysis of pine                         | TEA     | sens=3        |
| SI21       | US LNG GHG — process-model LCA (Mukherjee, gwp_ng source)     | LCA     | sens=7, tor=3 |

**Borderline** (low count, no tornado/MC — likely a short sensitivity section, confirm by opening):
- m21 / SI7 — pyrolysis vs Gasification-FT, waste tires (sens=3)
- m26 — gasoline/diesel via fast pyrolysis, design case (sens=2)

**Excluded** (keyword hit but no primary SA results):
- m28 Product & Process Design Principles — textbook (sens=28, oat=3: teaches the method)
- m34 / m35 — ISO 14040 / 14044 LCA standards (cite "sensitivity analysis" as a requirement)
- m1, m3, m6, m9 — reviews / landscape papers (passing mentions)

---

## Peer-study SA cross-check — what they perturbed, and how they set the range

**Run:** 2026-08-05 · read all 14 studies above (12 high-confidence + 2 borderline) · extract → reverse-track review · scope = variables that overlap our SA params.

**Method.** One extractor agent per PDF read the SA table (tables-first) via `pdftotext` and recorded every perturbed variable + baseline + range + how the range was set. A second, independent reviewer agent then **reverse-tracked** each matched row: followed the recorded location back into the PDF and confirmed the value/range is actually there. Per-(study × variable) record of truth = `sa/ref_check/peer_sa.json` (131 rows total; 41 overlap our params); raw agent JSON in `sa/ref_check/peer_raw/`.

**Headline: 41 matched rows across 13 studies · 40 CONFIRMED at cited location · 1 MISLOCATED (value correct, page off by one) · 0 value-mismatches. The dominant finding: peers set SA perturbation widths by author judgement ("reasonable ±X%"), NOT by citation — which directly backs our own un-sourced ±% ranges as standard practice.**

### Summary roll-up — one row per our SA param

| Our param | Our range | Peers that sweep it | Their ranges | How they set the range |
|---|---|---|---|---|
| feedstock_cost | ±50% | 12 (m5, m10, m23, m26, m27, m33, m21, SI1, SI4, SI5, SI8, SI11) | ±15% to ±30%; some min/max ($20–80/t, $35–80/t) | mostly **reasoning** (assumed/"reasonable min–max"); **cited** only in SI1 (refs 86,87), m5 (Davis 2018 + GREET 2022), SI5 (EIA, in SI) |
| ng_price | ±50% | 2 (SI11, SI5) | ±15% (SI11); distribution (SI5) | SI11 reasoning; SI5 cites U.S. EIA historical trends (SI Table S7) |
| catalyst_lifetime | 1–5 yr | 4 (m5, m26, m27, SI4) | 0.5–5 yr (m26), 6 mo–2 yr (m27), ±50% catalyst-loss (SI4), 0.1–5 g/kg loss (m5) | reasoning/design judgement; **cited** only in m5 (Dros 2015 + GREET) |
| discount_rate | 8–20% | 7 (SI1, SI4, SI5, m23, m26, m27, SI11) | ±15–20%, 0–30%, or fig-only | reasoning; SI5 cites prior studies (refs 35,66); m23/m27 span shown only in tornado figure |
| plant_life | 10–20 yr | 1 (SI1) | ±30% (20 yr base) | **cited** — refs 86,87 (prior studies); paper notes minimal effect |
| capex_mult | ±30% | 8 (m23, m26, m27, m33, SI1, SI4, SI8, SI11) | ±15%, ±20%, ±30%, −10/+40%, −50/+100% | reasoning ("capital-estimate accuracy band"); **cited** only in SI1 (refs 86,87) |
| gwp_ng | ±20% | 2 (SI21, SI5) | ±50% (CH₄ EF), ±20% (NG-combustion EF), distribution | SI21 reasoning ("selected by authors to represent a wide uncertainty range"); SI5 cites ecoinvent 3.8 + GREET 2020 |
| electricity_grid_factor | ±20% (grid↔renewable) | 2 (SI21, SI5) | 72–676 kgCO₂/MWh (three grid mixes); distribution | SI21 reasoning (three real grid mixes NG/US-avg/coal); SI5 cites ecoinvent/GREET |
| allocation_method | mass vs energy | 1 (SI5) | displacement vs energy allocation | reasoning — methodological choice, not a distribution |
| biomass_gwp | ±30% | **0** | — | **no peer study perturbs biomass GWP directly** |
| steam_ef | ±20% | **0** | — | **no peer study perturbs a steam EF** |
| cooling_ef | ±20% | **0** | — | **no peer study perturbs a cooling-water EF** |
| wwt_ef | ±20% | **0** | — | **no peer study perturbs a WWT EF** (our dominant LCA contributor — no peer precedent) |

### Detail — per (study × variable), how the range was set + where it lives

Verdict column: ✓ = CONFIRMED at cited location by reverse-track; ≈ = MISLOCATED (value correct, page off). Full quotes in `peer_sa.json`. **Our baseline** = the value our SA perturbs around (from `ref_map.json`, extracted from code — see per-param note for the code location).

**feedstock_cost** — *our baseline: **0.104286 $/kg** (BIOMASS_2016=0.06923 escalated ×CEPCI 1.50637; SA grid ×[0.50…1.50] = ±50%)*

| Study | Baseline | Range | How set | Location | ✓ |
|---|---|---|---|---|---|
| m5 (Batten DMCO) | $0.36/L contribution | $5/t increments; bounds in ESI S1–S2 | **citation** — Davis 2018 (ref 68) + GREET 2022 (ref 71) + eng. logic | p.1931 §3.3, Fig.7; ESI S1–S2 | ✓ |
| m10 (forest residues) | MSP $2.50/kg @10% IRR | $20–80/tonne | reasoning — "varies widely", no cite | p.4610 §3.2, Fig.5 | ✓ |
| m23 (jet cycloalkanes) | Table 2 feedstock price | fig-only span | reasoning — numeric span only in Fig.4 | p.296 §3.3, Fig.4 | ✓ |
| m26 (gasoline/diesel design) | ~$50/dry ton | $35–80/ton | reasoning — low/high endpoints, no cite | Fig.7.3 §7.3 | ✓ |
| m27 (NREL Jones 2013) | $60/dry ton (2011$) | up to ~$120/t (pine) → +14% | reasoning — pine as low-ash example | p.30 §5.3, Fig.11; base Table 14 | ✓ |
| m33 (catalytic deox+HTG) | nominal TRMC | ±30% | reasoning — OAT ±30%, no cite | p.12 §4.4.1, Fig.14 | ✓ |
| m21 (waste tires) | — | ±30% | reasoning — uniform ±30% tornado | p.10 §3.2.2, Fig.10 | ✓ |
| SI1 (BECCS) | not tabulated | ±30% | **citation** — "based on previous studies 86,87" | p.17 §3.10, Fig.8 | ✓ |
| SI4 (ethanol-to-jet) | $3.6/bushel; $80/dry ton | ±28% (grain); ±25% (stover) | reasoning — "reasonable minima/maxima" | p.1094 Table 9 | ✓ |
| SI5 (co-processing) | in SI | distribution | **citation/in-SI** — TEA studies/reports/patents (refs 35,48,62–65), Table S7 | p.8; Table S7 | ✓ |
| SI8 (HRJ) | $0.40/kg (jatropha) | min/max (Fig.7) | reasoning — single-point min/max, no % | p.13 Fig.7; base Table 3 | ✓ |
| SI11 (torrefaction pine) | $60/dry ton | ±15% | reasoning — uniform ±15%, no cite | p.4539; base Table 6 | ✓ |

**ng_price** — *our baseline: **0.054681 $/kg** (NG_2011=0.0363 = 76.57 ¢/MMBtu, escalated ×CEPCI 1.50637; SA grid ×[0.50…1.50] = ±50%)*

| Study | Baseline | Range | How set | Location | ✓ |
|---|---|---|---|---|---|
| SI11 | $5.04/GJ | ±15% | reasoning — uniform ±15% | p.4539; base Table 6 | ✓ |
| SI5 | in SI | distribution | **citation/in-SI** — U.S. EIA historical trends (refs 60,61), Table S7 | p.9; Table S7 | ✓ |

**catalyst_lifetime** — *our baseline: **3 yr** (COST.catalyst.year=3; SA grid [1 2 3 4 5])*

| Study | Baseline | Range | How set | Location | ✓ |
|---|---|---|---|---|---|
| m5 | 1 g loss/kg DMCO | 0.1 / 1 / 5 g/kg | **citation** — Dros 2015 (ref 67); make-up per GREET 2022 | p.1928 §2.4–2.5, Table 3 | ✓ |
| m26 | 1 yr | 0.5–5 yr | reasoning — bracket low/high, no cite | Fig.7.3 §7.3; base p.6.3 | ✓ |
| m27 | 1 yr/reactor | 6 mo (+9%) – 2 yr (−7%) | reasoning — plausible on-stream life | p.30 §5.3, Fig.11 | ✓ |
| SI4 | 348 / 625 lb·h⁻¹ | ±50% (catalyst-loss proxy) | reasoning — "reasonable min/max" | p.1094 Table 9 | ✓ |

**discount_rate** — *our baseline: **0.15** (COST.irate=0.15; SA grid [0.08 0.10 0.12 0.15 0.18 0.20])*

| Study | Baseline | Range | How set | Location | ✓ |
|---|---|---|---|---|---|
| SI1 | 10% | ±20% | reasoning — exception to ±30% set, no explicit basis | p.17 §3.10; base Table 4 | ✓ |
| SI11 | 10% IRR | ±15% | reasoning — uniform ±15% | p.4539; base Table 6 | ✓ |
| SI4 (IRR) | 40% (10% DCFROR) | −75/+150% | reasoning — "reasonable min/max" | p.1094 Table 9 | ✓ |
| SI4 (interest) | col. misaligned | ±50% | reasoning — "reasonable min/max" | p.1094 Table 9 | ✓ |
| SI5 | 10% IRR | distribution | citation/in-SI — prior studies (refs 35,66), Table S7 | p.8; Table S7 | ✓ |
| m23 | 10% IRR | fig-only | reasoning — span only in Fig.4 | p.296 §3.3, Fig.4 | ✓ |
| m26 | 10% ROI | 0–30% | reasoning — plant-gate to high-return endpoints | Fig.7.2 §7.2 | ✓ |
| m27 | 10% IRR | fig-only | reasoning — span only in Fig.11 | p.30 §5.3, Fig.11 | ✓ |

**plant_life** — *our baseline: **15 yr** (COST.plant_life=15; SA grid [10 12 15 18 20])*

| Study | Baseline | Range | How set | Location | ✓ |
|---|---|---|---|---|---|
| SI1 | 20 yr | ±30% | **citation** — refs 86,87; minimal effect noted | p.17 §3.10; base Table 4 | ✓ |

**capex_mult** — *our baseline: **1.0** (baseCOST.capex_mult=1.0 in SA_driver; SA grid [0.70 0.85 1.00 1.15 1.30] = ±30%)*

| Study | Baseline | Range | How set | Location | ✓ |
|---|---|---|---|---|---|
| SI1 | factorial + APEA (3.02) | ±30% | **citation** — refs 86,87 (→ ±15% cost swing) | p.17 §3.10; Table 5 | ✓ |
| SI11 | Peters et al. method | ±15% | reasoning — uniform ±15% | p.4539; Table 5 | ✓ |
| SI4 | col. misaligned | −50/+100% | reasoning — "reasonable min/max" | p.1094 Table 9 | ✓ |
| SI8 | $341–354M | min/max (Fig.7) | reasoning — single-point, no % | p.13 Fig.7; base p.12 | ✓ |
| m23 | $285.51M TCI | ±20% | reasoning — most-sensitive var, tornado | p.296 §3.3, Fig.4 | ✓ |
| m26 | $303M TPI | −10/+40% | reasoning — capital-estimate accuracy band | Fig.7.2 §7.2 | ✓ |
| m27 | $700M TCI | fig-only | reasoning — span only in Fig.11 | p.30 §5.3, Fig.11 | ✓ |
| m33 | nominal EPC | ±30% | reasoning — OAT ±30%, no cite | p.12 §4.4.1, Fig.14 | ✓ |

**gwp_ng / electricity_grid_factor / allocation_method (LCA)** — *our baselines: electricity_grid_factor **0.67420026 kg/kWh** (LCA.EL_GRID; ±20%) · allocation_method **mass-basis** (÷48.168; alt = energy-basis over SAF+gasoline LHV)*

| Study | Our param | Baseline | Range | How set | Location | ✓ |
|---|---|---|---|---|---|---|
| SI21 (Mukherjee LNG) | gwp_ng | — | ±50% (CH₄ EF onsite power) | reasoning — "wide uncertainty range, selected by authors" | p.4, Fig.2d | ✓ |
| SI21 | gwp_ng | — | ±20% (NG combustion, regas) | reasoning — "±20% where lack of data" | p.8 (extract said p.7), Fig.6c | ≈ |
| SI5 | gwp_ng | — | distribution | citation — ecoinvent 3.8 + GREET 2020 (Table S6/S7) | p.9; Table S6 | ✓ |
| SI21 | electricity_grid_factor | — | 72–676 kgCO₂/MWh | reasoning — three real grid mixes (NG/US-avg/coal) | p.12 Methods, Fig.6c | ✓ |
| SI5 | electricity_grid_factor | — | distribution | citation — ecoinvent/GREET (Table S6/S7) | p.9; Table S6 | ✓ |
| SI5 | allocation_method | displacement | displacement vs energy | reasoning — methodological choice | p.9 | ✓ |

> **Note — effective LCA feed factors (downstream multiplier applies).** In `TEA_LCA_non_CC_260407/LCA_analysis_SAF.m` the biomass and NG feed GWPs are **not** used at their bare emission-factor value: each base factor (hardcoded in `LCA_analysis_SAF.m`) is multiplied by a scaling constant from `LCA_PARAMETER.m` at lines 49–50. So the *effective* baseline that the LCA actually sums is:
>
> | Feed | Base factor (LCA_analysis_SAF.m) | Multiplier (LCA_PARAMETER.m) | **Effective** |
> |---|---|---|---|
> | Biomass | 0.1772 kgCO₂e/kg (`GWP_biomass`, L3) | `LCA.BM = 0.4` (L6) | **0.07088 kgCO₂e/kg** |
> | Natural gas | 0.2221 kgCO₂e/kg (`GWP_NG`, L7) | `LCA.NG_kg = 3` (L33) | **0.6663 kgCO₂e/kg** |
>
> Asymmetric: biomass scaled ×0.4, NG scaled ×3. The `gwp_ng` baseline cited above (0.2221) is the *base* constant, not the effective value the model uses. Confirm multipliers are intended before citing either number in the SA.

### Per-study summary + SI-to-fetch

| Code | Study (short) | Own SA? | # vars perturbed | # overlap with ours |
|---|---|---|---|---|
| m5 | Batten DMCO jet (Sust. Energy Fuels 2024) | yes | 6 | 2 |
| m10 | SAF from forest residues | yes | 7 | 1 |
| m23 | Lignocellulose → jet cycloalkanes | yes | 6 | 3 |
| m26 | Gasoline/diesel biomass fast pyrolysis (PNNL design) | yes | 12 | 4 |
| m27 | NREL Jones 2013 fast pyrolysis + HT | yes | 11 | 4 |
| m30 | CO₂ capture, phase-change solvent | yes | 10 | **0** (process/solvent SA) |
| m33 | SAF via catalytic deox + HTG | yes | 5 | 2 |
| m21 | Pyrolysis vs Gasification-FT, waste tires (borderline) | yes | 6 | 1 |
| SI1 | BECCS — FT/bioSNG/OMEx | yes | 7 | 3 |
| SI4 | Ethanol-to-jet blendstock upgrading | yes | 14 | 5 |
| SI5 | Co-processing SAF at refineries | yes (Monte Carlo) | 12 | 6 |
| SI8 | Hydroprocessed renewable jet (HRJ) | yes | 10 | 2 |
| SI11 | Torrefaction + fast pyrolysis of pine | yes | 6 | 4 |
| SI21 | Mukherjee US LNG GHG LCA | yes | 19 | 2 |

**SI to fetch later** (ranges/distributions deferred to Supplementary Information — main-text PDF insufficient): **SI5** (Table S7 parameter distributions; Table S6 LCI sources — covers feedstock, NG, grid CI, allocation) and **m5** (ESI Tables S1–S2 full numeric feedstock/catalyst bounds). Pull these SIs to get the exact swept min/max.

### Findings (most important first)

1. **Peers set SA range widths by judgement, not citation — this backs our un-sourced ranges.** Of the 41 matched rows, only ~6 cite any source for the *perturbation width*: SI1 (±30% "based on previous studies 86,87"), m5 (feedstock bounds from Davis 2018 + GREET 2022; catalyst from Dros 2015), and SI5 (distributions from EIA / ecoinvent / GREET, in the SI). The rest — SI11 (±15%), m21 (±30%), m33 (±30%), SI4 ("reasonable minima and maxima"), m10, m26, m27, SI8 — set the width by author judgement with **no reference**. So our own un-sourced sweeps (catalyst_lifetime, discount_rate, plant_life, capex_mult, and the ±20% LCA EFs) are squarely standard practice; the reviewer response can say so and cite these papers as precedent.
2. **±30% is the modal width; our ranges sit inside the peer envelope.** ±30% recurs (SI1, m21, m33, capex); ±15–20% is the other common band. Our capex ±30% and EF/GWP ±20% match peers directly; our feedstock ±50% and discount 8–20% are at/near the wide end but within range.
3. **No peer study perturbs our LCA emission factors (steam_ef, cooling_ef, wwt_ef, biomass_gwp).** Peers doing LCA-SA vary CH₄/NG-combustion EFs and grid CI (SI21) or bundle LCI uncertainty into Monte-Carlo distributions (SI5). Our steam/cooling/**WWT** (WWT = our dominant LCA contributor) and biomass-GWP sweeps therefore have **no direct peer precedent** — their ±20/30% widths are unavoidably our own engineering assumption (defensible as generic LCI uncertainty, but not attributable to a peer SA).
4. **Variables peers sweep that we don't — candidates we may be missing.** Recurrent high-impact peer SA variables absent from ours: **product/fuel yield & conversion** (SI11 calls biofuel yield its #1 driver; m27 product yield; m10 FT conversion), **plant scale/capacity** (SI8, m26, m27), **H₂ price** (SI8, m21, m23 — which we *discarded*), tax rate (m21, m33), and co-product/H₂ credit price (m23, SI4). Yield and plant scale in particular are top MSP drivers in several TEAs.
5. **Reverse-track validation is clean.** 40/41 rows CONFIRMED at their cited location; the one exception (SI21 NG-combustion EF ±20%) is MISLOCATED only — value correct, quote on p.8 not p.7. No value mismatches, giving high confidence in the extracted ranges.
6. **Figure-only ranges (need a figure read to get exact min/max).** For m23, m27 (feedstock, IRR, TCI) the swept low/high appear only inside tornado-chart images, which `pdftotext` can't read. The variable + baseline are confirmed; the exact numeric span would need the `pdftoppm`/figure-read fallback (see TODO). m5 and SI5 exact spans are in their SIs (above).
7. **Borderline studies resolved.** m21 (waste tires) has a genuine ±30% tornado SA — keep. m26 (PNNL design case) has a full research-sensitivity set (feedstock, catalyst life, ROI, TPI + 8 process vars) — a strong match, keep. m30 (CO₂ solvent) swept 10 solvent/process variables with **zero** economic/LCA overlap — correctly a process-design SA, not a parameter SA; exclude from the cross-comparison.
