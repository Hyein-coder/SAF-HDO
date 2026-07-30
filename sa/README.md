# Additional TEA sensitivity analysis (CEJ reviewer response)

Answers CEJ reviewer requests for a techno-economic sensitivity analysis of
**MSP** (minimum selling price of SAF) w.r.t. **catalyst lifetime**,
**hydrogen price**, **feedstock/pyrolysis-oil cost**, and general robustness.

## Why this is separate from the existing SA
`Sensitivity_1210/` and `to_Nurul_20260209/` sweep the *process/product
distribution* (Peak Shift, Heavy-end, reaction coefficients) → MSP/GWP/H2.
The reviewers ask for a *cost-parameter* SA, which did not exist. Because the
Aspen mass/energy balances are fixed, this SA reuses the frozen converged
results (`processData`) and only re-runs the MATLAB cash-flow model — **no
Aspen re-run needed**.

## Files
| File | Role |
|---|---|
| `SA_driver.m` | main entry point; loads base case, sweeps parameters, writes CSVs |
| `cash_flow_SA.m` | SA variant of `cash_flow_generic_SAF_opt.m` (adds catalyst replacement, CAPEX multiplier, parametric catalyst cost) |
| `msp_npv_SA.m` | returns NPV(x) for the MSP root-solve (`fzero`, base MATLAB only) |
| `plot_sa.py` | renders `tornado.png` + `spider.png` from the CSVs (numpy+matplotlib) |

## How to run
Headless (MATLAB R2025b/R2026a are installed and on PATH):
```
matlab -batch "cd('D:\SAF_Nurul\saf_nurul_agent\sa'); SA_driver"
python plot_sa.py
```
From Python via the MATLAB Engine API (optional; not installed by default):
```
pip install matlabengine==26.1.*      # match R2026a; or install from matlabroot\extern\engines\python
python -c "import matlab.engine; e=matlab.engine.start_matlab(); e.cd(r'D:\SAF_Nurul\saf_nurul_agent\sa'); e.SA_driver(nargout=0)"
```
`matlab -batch` is recommended — no extra install, fully scriptable.

## Modeling decision (resolved)
The **published** cash-flow model charges **no catalyst replacement cost**:
`COST.catalyst.year` (=3 yr) and `COST.catalyst.DECHLORIN/ETP1/ETP2` ($/yr)
are defined but never used, and the `mod(...,catalyst.year)` branch in
`cash_flow_generic_SAF.m:160` is a no-op (both branches identical). So in the
as-published model **catalyst lifetime has zero effect on MSP**.

**Decision (SA only, base model left unchanged):** `cash_flow_SA.m` charges
catalyst replacement as **annualized OPEX = `C_inventory / catalyst.year`**
(`C_inventory = $59,707.35`, the value already in the code), every operating
year (`COST.catalyst.replace = true`). Shorter lifetime → higher annual cost →
higher MSP, so lifetime is a live lever. The published base MSP is **not**
changed; `SA_driver.m` reports **both** the published base MSP and the SA
baseline (with replacement) so the delta is explicit.

## Parameters and ranges
| Parameter | Base | Range | Basis |
|---|---|---|---|
| Hydrogen price | 1.3 $/kg | 1.0–3.34 $/kg | grey↔green literature spread (3.34 already cited in COST_PARAMETER) |
| Feedstock cost | ~0.11 $/kg ($80/dry-ton) | ±50% | NREL feedstock, reviewer "feedstock variation" |
| Catalyst lifetime | 3 yr | 1–5 yr | reviewer request |
| Discount rate | 0.15 | 0.08–0.20 | standard TEA range |
| Plant life | 15 yr | 10–20 yr | standard TEA range |
| CAPEX | ×1.0 | ×0.7–1.3 | ±30% cost-estimate uncertainty |
| Natural gas price | escalated NREL | ±50% | utility uncertainty |

## Outputs
- `sa_oat.csv` — one-at-a-time sweeps (parameter, value, MSP) → spider plot
- `sa_tornado.csv` — low/high MSP per parameter → tornado chart
- `tornado.png`, `spider.png`

## Phase 2 (not yet wired) — GWP / LCA sensitivity
Reviewer 2 ties reliability to the LCA. A GWP sensitivity for **H2 source**
(`LCA.H2_GREY`=1.579 vs `LCA.H2_GREEN`=0) and **feedstock carbon intensity**
should follow, reusing `LCA_analysis_SAF.m`. Deferred until that function's
inputs are reviewed so we don't ship untested LCA numbers.
