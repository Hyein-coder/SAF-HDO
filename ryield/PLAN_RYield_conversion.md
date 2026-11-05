# Plan: convert cases i / j / k HDO section to RYield

Date: 2026-10-20 · Status: **plan only — nothing modified yet, awaiting decisions (§5)**

## 1. Goal
Replace the 3-stage RStoic HDO train in the final-draft case files (i, j, k) with an RYield-based HDO
section, following the pattern of the reference `D:\SAF_Nurul\250717_pyrolysis_oil_CC_RYield.bkp`.

## 2. Inputs

| Role | File |
|---|---|
| Reference (RYield pattern) | `D:\SAF_Nurul\250717_pyrolysis_oil_CC_RYield.bkp` |
| Case k (C1) | `D:\SAF_Nurul\res_20260403_non_CC\res_20260403_non_CC\260401_pyrolysis_oil_CC_case_k_6_non_CC.bkp` |
| Case j (C2) | `...\260401_pyrolysis_oil_CC_case_j_HOadjustment3_non_CC.bkp` |
| Case i (B4) | `...\260120_pyrolysis_oil_CC_case_i_non_CC.bkp` |
| TEA/LCA code | `...\res_20260403_non_CC\TEA_LCA_non_CC_260407\` (entry: `Sensitivity_analysis_SAF.m`) |

Baseline MSP to compare against ($/kg): j 1.0847, k 0.9496, i 1.4061.

## 3. Findings

### 3.1 Reference RYield file
- HDO = **single RYield `R-201`**, 350 °C / 100 bar (`SPEC-OPT = TP`).
- Yields on **mass basis** (`BASIS = MASS`): CH4, C6–C30 hydrocarbons, WATER (~30 products). E.g. CH4 0.2489,
  WATER 0.2011, C9H18 0.0808, C8H16-CY 0.0763 … C30H62 3.3E-08.
- Topology:
  ```
  206 ─► B-A (Sep) ─► SA ─► R-201 (RYield) ─► 207 ─┐
                   └─► SB (bypass) ─────────────────┴─► B-B (Mixer) ─► SC ─► E-202 (Heater) ─► 208
  ```
- `B-A` spec: every component FRACS = 1 to SA, **except H2 = fixed flow 6149.847** (consumed H2).
  The excess H2 goes around the reactor in SB. This keeps RYield mass balance closed (RYield normalizes product
  mass to its inlet mass).
- **Weakness:** inert/non-reacting species (CO, CO2, N2, AR, O2, SO2, NO2, ASH, SIO2 …) are also sent to SA, so
  RYield converts their mass into products. Should be routed to SB instead.

### 3.2 Current case files (i / j / k)
- HDO = 3 RStoic in series, `SERIES = YES`:

  | Block | T | P | # rxns i / j / k |
  |---|---|---|---|
  | R-201 | 200 °C | 100 bar | 55 / 63 / 65 |
  | R-202 | 355 °C | 100 bar | 7 / 8 / 10 |
  | R-203 | 350 °C | ΔP 0 | 5 / 5 / 5 |

- Surrounding blocks: `M-202` (147+205→206), `E-202` HeatX (206/214→207/215), `R-201` (207→208),
  `E-203` HeatX (208/213→209/214), `E-204` (209→210), `R-202` (210→211), `E-205` (211→212),
  `R-203` (212→213), `E-206` heater (215→216).
- Downstream stream names differ from reference (reference continues as SC→208; cases continue 215→216).

### 3.3 MATLAB TEA dependencies
- `accessData_SAF.m` enumerates **all** blocks generically; `CAPEX_SAF.m` costs by block **type**
  (`RStoic` l.195, `RYield` l.258, `Sep` l.17). No HDO block is referenced by name. Removing R-202/R-203 does not break code.
- Hard-coded streams: `206` and `215` (H2 consumption = 206 H2 − 215 H2; CH4 from 215) in
  `accessData_SAF.m:170`, `Sensitivity_analysis_SAF.m:65-95`, `SA_min_data.m`, `tessasaf.m`.
  **Keep stream names 206 and 215 in the new files, so MATLAB needs no changes.**
- Unnamed `Sep` blocks (B-A) fall through with no cost (B-A branch is commented out in CAPEX), which is intended.

## 4. Plan

1. **Copy, don't edit the archive.** Copy the 3 bundle `.bkp` files to the repo as
   `YYMMDD_pyrolysis_oil_CC_case_{i,j_HOadjustment3,k_6}_non_CC_RYield.bkp`. `D:\SAF_Nurul` stays read-only.
2. **Derive yield vector per case** (script, via Aspen COM on converged current case):
   - Reactor inlet = stream 206 minus excess H2 (excess = 215 H2 after train).
   - Mass yield_c = (outlet-of-R-203 flow_c) / (reactor inlet total mass), for hydrocarbons, water, light gases.
   - B-A H2 fixed flow = 206 H2 − 215 H2 (same quantity MATLAB uses as H2 consumption).
   - Output: one yield table per case (CSV), saved alongside the new `.bkp`.
3. **Change the flowsheet** (per case):
   - Delete `R-202`, `R-203`, `E-203`, `E-204`, `E-205` and the stream links between them.
   - Replace `R-201` RStoic with RYield; add `B-A` (Sep) and `B-B` (Mixer) as in the reference.
   - Wire: 206 → [preheat] → B-A → SA → R-201 → B-B ← SB; B-B outlet = **215** → E-206 → 216 (unchanged downstream).
   - Topology: recommend a one-time manual edit in the Aspen GUI. Script fills yields and B-A specs via COM.
4. **Fix the inert-bypass issue:** in B-A, send non-reacting species (CO, CO2, N2, AR, O2, SO2, NO2, ASH, SIO2,
   and any species absent from the yield list) to SB, not SA.
5. **Conditions and heat integration:** R-201 at 350 °C / 100 bar (reference). Single reactor removes the
   E-202/E-203 feed–effluent HeatX pair, so add one feed–effluent HeatX or a heater so utility duties stay comparable.
6. **Validate:**
   - Run each new `.bkp`; check RYield mass balance and that no inert disappears.
   - Downstream products (SAF, gasoline, H2 stream 419) should match the RStoic case within ~0.1 % (if decision A = a).
   - Rerun MATLAB `Sensitivity_analysis_SAF.m` per case; tabulate MSP / GWP vs. baseline above. Expect differences
     only from CAPEX (3→1 reactor) and heat duties.
7. **(Optional) BO refit:** `aspen_utils.py` writes `\Data\Blocks\R-201\Input\CONV` (RStoic). For RYield, write
   `\Data\Blocks\R-201\Input\MOLE_YIELD\MIXED\<comp>` instead (verify exact node path in Aspen V14 variable explorer).

## 5. Decisions needed

| # | Decision | Options | Recommendation |
|---|---|---|---|
| A | **Yield source** | (a) from current RStoic results, so i/j/k are reproduced exactly · (b) from experimental carbon-number distribution (`HDO_exp.xlsx`), like the reference's smooth C6–C30 curve, so products change | (a) |
| B | **Reactor count / CAPEX** | (a) one lumped RYield (reference): lower HDO CAPEX, MSP shifts · (b) keep 3 vessels for costing (e.g. RYield + 2 dummy vessels / CAPEX override) | Need your call; depends on why RYield is wanted (reviewer response?) |
| C | **Topology edit method** | (a) manual GUI edit once per case, script does the rest · (b) fully scripted via COM (`Elements.Add("R-201!RYIELD")`, port connects) | (a) |
| D | **Inert handling** | (a) bypass inerts via B-A (fix) · (b) copy reference as-is | (a) |
| E | **Heat integration** | (a) one feed–effluent HeatX · (b) plain heater (as reference E-202) | (a) |
| F | **Output location / naming** | e.g. `D:\saf_hdo\aspen\RYield_YYMMDD\`, suffix `_RYield` | confirm |

## 6. Risks
- RYield does not enforce atom balance. Check H/C/O element balance of each yield set (especially option A-b).
- Fixed H2 flow in B-A goes stale if the feed changes (e.g. SA sweeps on feed rate). Consider a Calculator
  that sets B-A H2 = yield-implied consumption × feed.
- Aspen V14 version/upgrade prompts when opening 2025-07 reference; open copies only.
