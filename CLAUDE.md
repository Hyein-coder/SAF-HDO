# CLAUDE.md

## Workspace & access scope

- **Working directory:** `D:\saf_hdo` — do all new/scratch work **here**. This git repo holds the active copies of the Python BO + Aspen driver code, plotting/analysis scripts, and results. Create files, scripts, notes, and outputs in this repo.
- **Reference directory:** `D:\SAF_Nurul` — the full research archive. **Read it freely for reference** (data, code, simulations, manuscripts), but **do not modify anything under it** unless the user explicitly asks. Treat it as a read-only source of truth (harness permissions deny Write/Edit there).
- Paths use Windows separators. The Bash tool is available (Git Bash / POSIX) alongside PowerShell — use forward slashes or quote Windows paths in Bash.

### Notion access scope (firm)

- **Write/edit only** within the **Dashboard** page and its descendants — `https://app.notion.com/p/3adf9a841fe1805582d5fa6c429e58ad` (incl. the nested **Tasks** database `collection://08879e7e-e5f0-40bb-b869-82bb39f8b57c` and its views).
- **Everything else** in the KIST Pico-to-Giga Lab workspace is **read-only.** No create/update/move/delete/comment outside the Dashboard subtree without an explicit one-time override from the user.
- Before any Notion write, confirm the target resolves inside the Dashboard subtree; if not, refuse and ask. (Harness permissions can't page-scope Notion MCP tools — this rule is the enforcement.)

## What this project is

Techno-economic analysis (**TEA**) and life-cycle assessment (**LCA**) of **Sustainable Aviation Fuel (SAF)** produced from **biomass fast pyrolysis → hydrodeoxygenation (HDO)**, with optional **carbon capture (CC)**. Work spans process simulation, reaction-coefficient fitting, economic/environmental modeling, sensitivity analysis, and manuscript writing (journal submission in progress). Institution: KIST.

Process chain: biomass → fast pyrolysis → pyrolysis oil → 3-stage HDO reactors → fractionation → SAF (+ gasoline, H₂ recovery), with a hydrogen-production section and CO₂ capture.

## Toolchain

1. **Aspen Plus** — process simulations in `*.bkp` / `*.apw` / `*.appdf` files (named `*_pyrolysis_oil_CC_case_*`). Binary; driven programmatically via COM (`Apwn.Document`).
2. **Python** (`main_251001/`) — Bayesian optimization that fits HDO reaction stoichiometry to experimental carbon-number distributions:
   - `aspen_utils.py` — `AspenSim` class; drives Aspen via `win32com`, reads/writes reactor conversion nodes (`\Data\Blocks\R-201\Input\CONV`), reads product-stream composition. Component list + `component_to_carbon_number` map live here.
   - `main.py` — GPyOpt `ModularBayesianOptimization` loop minimizing MSE between simulated and experimental (`HDO_exp.xlsx`) carbon-number composition; saves optimal reaction coefficients and plots.
   - `utils.py` — `CustomEvaluator` wrapper. `configuration_ver250330_jisung_v2.py` — compressor/vessel node config.
3. **MATLAB** (`Create_251104_2/Create_251104_2/`) — the TEA/LCA engine, invoked from `MainTest_SAF.m`:
   - `callaspen.m` / `accessData_SAF.m` — pull stream & unit data from an Aspen `.bkp` via ActiveX.
   - `CAPEX_SAF*.m`, `OPEX_SAF.m`, `NPV_*_2020.m` (per-equipment cost correlations), `cash_flow_generic_SAF*.m`, `msp_SAF.m` — minimum selling price (**MSP**).
   - `LCA_analysis_SAF.m`, `LCA_PARAMETER.m`, `estimate_CO2_capture_cost.m`.
   - `COST_PARAMETER.m` / `COST_PARAMETER_251110.txt` — economic assumptions (plant life 15 yr, tax 0.3892, interest 0.15, CEPCI escalation 2011→2022, feedstock/NG/H₂/SAF prices with literature sources, MACRS depreciation).
   - `main_rxn_coeff.m` (in parent root) — MATLAB variant of the reaction-coefficient fit.

## Conventions

- **Case labels** `a`, `f`, `i`, `j`, `k` denote distinct process configurations (with/without CC, different mechanisms). `non_CC` = carbon-capture removed.
- **Filenames are date-stamped** `YYMMDD_...` (e.g. `260115_...` = 2026-01-15) and versioned (`Rev1`…`Rev13`). The **highest Rev / latest date is the current version.** `$backup`, `(AutoRecovered)`, `.asv` are stale — ignore unless asked.
- Manuscript = `SAF_*.docx` / `SAF_HDO_TEALCA_v6_*.docx`; SI = `Supporting Information_SAF_*.docx` (contents listed in `SI_listofcontents.txt`); references in `20260304_Nurul_manuscript_and_references/*.bib`.

## Where things live (reference archive `D:\SAF_Nurul`)

| Area | Location |
|---|---|
| Python BO + Aspen driver | `main_251001/` |
| MATLAB TEA/LCA | `Create_251104_2/Create_251104_2/` |
| Latest simulations | `20260330_NewSim/`, `Simulation_260126/` |
| SAF-BJF case bundle | `20260227_Nurul_SAF-BJF/` |
| Sensitivity-analysis figures/data | `Sensitivity_1210/`, `to_Nurul_20260209/` |
| Manuscript + refs | `Manuscript/`, `20260304_Nurul_manuscript_and_references/`, `20260423_Nurul_feedback/` |
| Experimental data | `HDO_exp.xlsx`, `Economic Data_250918.xlsx`, `Pyrolysis Oil_*.xlsx` |

## Notes for working here

- **Python env:** use the conda env **`saf_hdo`** (`conda run -n saf_hdo python ...`) — it has `pandas`, `scipy`, `openpyxl` (Excel), etc. The `base` env lacks `openpyxl` so `pd.read_excel` fails there. Don't `pip install` into `base`; if a package is missing, add it to `saf_hdo` (or clone it) rather than polluting `base`.
- Aspen `.bkp`/`.apw` and Office `.docx`/`.pptx`/`.xlsx` are **binary** — don't cat them; extract via the appropriate tool/library or ask the user. `.zip` archives usually duplicate an adjacent unzipped folder.
- Many hard-coded absolute paths in the code (`D:\SAF-HDO py\...`, `D:\523506\...`) point to the author's original machine — verify/repoint before running.
- When writing manuscript content, a `kan-manuscript` skill exists but targets a **different** (KAN sensitivity-analysis) paper — do not use it for SAF work unless the user asks.
