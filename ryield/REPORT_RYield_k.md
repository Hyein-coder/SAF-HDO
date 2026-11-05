# Case k HDO: RStoic → RYield conversion

Session report · 2026-10-20 to 2026-10-22 · branch `sa_2nd`

## Summary

The three RStoic HDO reactors (R-201, R-202, R-203) in case k were replaced with RYield blocks. Each RYield's yields come from the converged RStoic results. The new file reproduces the original at the base operating point: reactor and heat-exchanger duties agree to about 1e-6 relative, and the HDO effluent agrees to 1.6e-6. GPT Codex reviewed the work twice. One conversion bug was found and fixed. Fixed yields only hold at this operating point (see Limitations). That doesn't affect the current TEA/LCA workflow, which never changes Aspen inputs.

## 1. Topology comparison

The RYield file `D:\SAF_Nurul\250717_pyrolysis_oil_CC_RYield.bkp` was compared with the final-draft case files in `res_20260403_non_CC`. No `TEA_LCA_non_CC_250731_*` directory exists. The closest match, `TEA_LCA_non_CC_260731_catOPEX`, contains no `.bkp` files, and its input data comes from the 260403 bundle.

- Cases i, j and k have identical topology: 81 blocks and 122 streams each.
- The 250717 RYield file has 78 blocks and 119 streams. It differs only in the HDO section and one extra block:

| | 250717 RYield file | Cases i / j / k |
|---|---|---|
| HDO reactors | one RYield (R-201) | three RStoic in series (R-201 at 200 °C, R-202 at 355 °C, R-203 at 350 °C) |
| H₂ handling | B-A (SEP) sends all oil plus a fixed 6,150 kg/h of H₂ to the reactor; excess H₂ bypasses it and B-B (MIXER) remixes | no bypass |
| Heat integration | E-202 plain heater | E-202 and E-203 heat exchangers (reactor effluent heats the feed); heaters E-204, E-205, E-206 |
| Other | S-402 SEP collects streams 107, 142, 330 and 424 | those streams leave as products |

## 2. Decisions (user)

| Question | Choice |
|---|---|
| What the RYield yields should match | the converged case-k RStoic outlet, so products stay unchanged |
| Reactor structure | one RYield per stage, so topology and block names stay the same |
| Reviewer | GPT Codex CLI as critic |

Source file: `D:\SAF_Nurul\res_20260403_non_CC\res_20260403_non_CC\260401_pyrolysis_oil_CC_case_k_6_non_CC.bkp` (read only).

## 3. Method

1. **Baseline** (`snapshot.py`): run the source through COM and record, for all 122 streams, the component mass flows, T and P. Also record each block's B_TEMP, B_PRES, QCALC, HX_DUTY and the other duty values.
2. **Yields** (`convert_k.py`): for each stage, yield of component *i* = outlet mass flow of *i* ÷ total inlet mass flow, on a mass basis.

   | Stage | Inlet → outlet | Number of yields | Σout / Σin |
   |---|---|---|---|
   | R-201 | 207 → 208 | 48 | 0.9999999985 |
   | R-202 | 210 → 211 | 46 | 1.0000000023 |
   | R-203 | 212 → 213 | 42 | 0.9999999998 |

   Each stage's mass balance closes, so RYield's mass normalization doesn't distort the yields.
3. **Swap blocks** by editing a text copy of the `.bkp`:
   - change `BLKTYPE`/`MDLTYPE` in the flowsheet section;
   - rebuild each `? BLOCK RSTOIC` section as `RYIELD`, keeping the original TEMP and PRES, adding `SPEC-OPT = TP`, writing the MOLE-YIELD rows with `BASIS = MASS`, and keeping the original `PRODUCTS` and display-settings lines at the end of the section.

   Editing the text was chosen over COM because RYield yields are a two-dimensional list that is awkward to fill through COM.
4. **Validate** (`compare.py`): run the new file and diff every stream and every block value. A value is flagged when the absolute difference exceeds 1e-6 and the relative difference exceeds 1e-4.

## 4. Bug found and fixed

In the first attempt, R-201's section header wraps onto two lines in the source file. Cutting at the first newline dropped `ICON1 ;`. R-201 then did nothing: QCALC = 0 and its outlet equalled its inlet.

Side observation: R-202's RYield still produced the exact R-202 outlet and took on R-201's duty (−3.81 MW). This shows that an RYield outlet does not depend on what flows in. That property drives the limitation in §8.

Fix: cut the header at `\ PARAM` instead of at the first newline.

## 5. Results (after the fix)

| Item | RStoic | RYield |
|---|---|---|
| R-201 duty (W) | −3,669,071.5 | −3,669,075.9 |
| R-202 duty (W) | −371,535.2 | −371,534.6 |
| R-203 duty (W) | −157,204.9 | −157,204.6 |
| E-202 / E-203 HX duty (W) | 1,658,956 / 960,781 | 1,658,956 / 960,779 |
| E-206 duty (W) | −1,278,538 | −1,278,533 |
| HDO effluent 216 (kg/h) | 29,646.852 | 29,646.804 |
| H₂ recycle 419 (kg/h) | 1,701.528 | 1,701.531 |
| Largest temperature difference | | 0.015 °C |
| Run status | warning on R-101 only | same (the warning was already there) |

58 values remain flagged, all in recycle and column side streams under 11 kg/h (sections 300 and 400) and in the N2COMPOU recycle. The largest absolute difference is 0.71 kg/h of N₂ in the combustion-air streams 328–334. The CM-301 condenser duty moves by 63 W. These look like recycle and design-spec convergence tolerance, not a reactor difference. That hasn't been proven with a tighter-tolerance run.

## 6. TEA/LCA impact

- **Cost code:** in `CAPEX_SAF.m`, the RStoic branch (l.195) and the RYield branch (l.258) are identical. Both call `NPV_CSTR_2020`, sized on outlet mass flow, density and B_PRES. Reactor CAPEX is therefore unchanged.
- **Values the TEA reads:** `accessData_SAF.m` reads B_TEMP, B_PRES, QCALC/HX_DUTY and stream flows. H₂ consumption is taken as stream 206 H₂ − stream 215 H₂. All of these match.
- **SA scripts:** `SA_min_data.m`, `SA_LCA.m` and `LCA_OAT_SAF.m` only re-run fixed files and never change Aspen inputs.

The expected MSP is unchanged (0.9496 $/kg), but the MATLAB TEA hasn't been re-run to confirm it.

## 7. Codex review

Codex was run with `codex exec -s read-only`. On Windows the read-only sandbox blocks file access, so round 1 was generic. Round 2 had the evidence pasted into the prompt.

| Round | Main points | Response |
|---|---|---|
| 1 (method) | check that each stage's yields sum to 1; show Aspen really reads the blocks as RYield; use absolute plus relative tolerances; check how the TEA costs reactors by block type; fixed yields fail off-design | closure checked; Aspen ran the blocks and their duties match; `compare.py` uses both tolerances; CAPEX branches confirmed identical |
| 2 (results) | describe the result as agreement at the base point; test with tighter tolerances; the converter is fragile (assumes a following `? BLOCK`, `PRODUCTS` marker, line wrapping); test one off-design change | wording adopted. The E-notation and line-wrapping concerns are settled because Aspen parsed and ran the file. Codex misread the CM-301 shift as 63 kW; it is 63 W. |

## 8. Limitations

- **The yields only hold at the base point.** RYield gives the same product composition whatever the inlet, so changes to feed rate, H₂ excess or feed composition won't change the chemistry. Excess H₂ and inert components are included as yields, unlike the 250717 bypass design.
- **Reaction detail is gone.** Reaction stoichiometry and conversion are no longer in the model, so the Bayesian-optimization code in `aspen_utils.py` (which writes `\Input\CONV`) doesn't apply to these blocks.
- **The converter is case-specific.** `convert_k.py` assumes the section layout of these case files. It is not a general `.bkp` editor.

## 9. Files (`ryield/`, not committed)

| File | Purpose |
|---|---|
| `261022_pyrolysis_oil_CC_case_k_6_non_CC_RYield.bkp` | converted case k |
| `snapshot.py` | run a `.bkp` and dump stream and block results to JSON |
| `convert_k.py` | compute the yields and swap RStoic to RYield |
| `compare.py` | diff two snapshots |
| `base_k_rstoic.json`, `new_k_ryield.json` | baseline and converted snapshots |
| `probe_nodes.py` | one-off listing of COM node names |
| `validate_k.py`, `val_k_rstoic.json`, `val_k_ryield.json`, `VALIDATION_k_tables.md` | key-stream, phase-fraction and experimental-validation comparison (§10) |
| `PLAN_RYield_conversion.md` | earlier plan (single-reactor layout); replaced by the per-stage decision |

Environment: Aspen Plus V14. Python with pywin32 from the hermes venv; the `saf_hdo` conda env was not found on this machine.

## 10. Process and experimental-validation comparison (added 2026-10-22)

`validate_k.py` runs each file through `aspen_utils.AspenSim` with `rxtor_nodes=[]`, so the reaction-coefficient perturbation isn't applied and the reactor inputs aren't touched. It reuses the validation criteria from `drHa_rxn_tuning.py`. Full tables are in `VALIDATION_k_tables.md`. Both runs report the same warning on R-101.

**Mass flows (kg/h).** Every listed stream agrees within 5.3e-6 relative.

| Item | Stream | RStoic | RYield |
|---|---|---|---|
| Biomass feed, wet (dry biomass, NC) | 103 | 59,140.28 (41,283.26) | identical |
| Fresh H₂ feed | 201 | 1,691.53 | 1,691.53 |
| H₂ to HDO (fresh + recycle) | 205 | 3,000.05 | 3,000.05 |
| Pyrolysis oil to HDO | 146 | 26,646.80 | 26,646.80 |
| R-201 / R-202 / R-203 outlet (total) | 208 / 211 / 213 | 29,646.85 each | 29,646.85 / 29,646.80 / 29,646.80 |
| H₂ remaining after R-201 / R-202 / R-203 | 208 / 211 / 213 | 1,533.37 / 1,382.21 / 1,321.13 | same |
| H₂ consumed in HDO (207 − 216) | | 1,678.87 | 1,678.87 |
| SAF | 326 | 8,307.08 | 8,307.07 |
| Gasoline | 324 | 1,110.04 | 1,110.03 |
| Diesel-range heavy cut, burned in R-301 (case k has no diesel product) | 321 | 4,297.88 | 4,297.87 |
| CO₂ emitted (sum of 107, 142, 330, 424) | | 55,742.99 | 55,742.98 |

**Phase fractions.** Vapour and liquid fractions are identical to five decimals in all 18 streams. The ones that matter:
- R-201 inlet (207): VF 0.94304.
- R-201 outlet (208), two-phase at 200 °C: VF 0.78912.
- R-202 and R-203 outlets (211, 213): all vapour.
- HDO effluent (216) at 25 °C: VF 0.51334.
- Aqueous phase (301), oil (302) and all products: all liquid.

**Experimental match.** The two models give the same values.

| Metric | Experiment, case k | Experiment, bench | RStoic = RYield |
|---|---|---|---|
| HDO oil (302) H/C, molar | 1.8809 | 1.9856 | 2.1340 |
| HDO oil (302) O/C, molar | 0.0072 | 0.0040 | 0.0073 |
| Liquid yield, (302 + 301) / 216 | — | 0.841 | 0.9216 |
| Oil yield, 302 / 216 | — | 0.421 | 0.4770 |
| Aqueous fraction, 301 / 216 | — | — | 0.4445 |
| Carbon-number distribution of 216 vs `HDO_exp.xlsx` column k | | | MAE 0.00330 (models differ by at most 1.1e-10) |
| Pyrolysis oil (146), dry basis C / H / O, mass | — | — | 0.622 / 0.068 / 0.308; water 0.235 |

- **No regression from RYield.** The conversion leaves the match to experiment unchanged.
- **O/C and the carbon-number distribution match well.**
- **H/C is about 13% above the case-k target.** This mismatch was already in the RStoic model.
- **The liquid and oil yields exceed the bench values.** Neither yield has a case-k experimental value; the bench values are the only reference.
- **The pyrolysis oil has no experimental target in the existing code.** `find_pyro.py` only computes these values. R-101 is unchanged, so the numbers are identical by construction.

## 11. Next steps

1. Convert cases i and j with `convert_k.py`. The stage streams are the same, but the script hasn't been run on them.
2. Re-run the MATLAB TEA on the RYield k file and confirm MSP 0.9496 $/kg.
3. Optional: re-run both files with tighter tear-stream and design-spec tolerances to confirm the residual differences are noise.
4. Before any SA that changes Aspen inputs, change one feed variable in both models and compare SAF output and H₂ demand, or add an H₂/inert bypass.
