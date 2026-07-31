# TEA sensitivity — cases i, j, k (catalyst OPEX included in COM)

TEA version: `TEA_LCA_non_CC_260731_catOPEX` (catalyst replacement charged as
annualized OPEX = C_inventory / lifetime, in `C_util`/`COM`). Frozen Aspen
results reused (no re-run); cash-flow ported from `sa/cash_flow_SA.m` and
validated to reproduce each published MSP to 0.000%.

## Base MSP ($/kg)
| case | published (no catalyst OPEX) | this TEA (catalyst replacement) | Δ |
|---|---|---|---|
| i | 1.40612 | 1.40661 | +0.00049 |
| j | 1.08465 | 1.08505 | +0.00040 |
| k | 0.94957 | 0.94992 | +0.00035 |

Catalyst OPEX raises MSP by ~0.03–0.04 % — negligible.

## Tornado swing (|MSP_high − MSP_low|, $/kg), ranked
| parameter | range | i | j | k |
|---|---|---|---|---|
| Feedstock cost | ×0.5–1.5 | 0.846 | 0.698 | 0.610 |
| Discount rate | 0.08–0.20 | 0.247 | 0.202 | 0.180 |
| CAPEX multiplier | ×0.7–1.3 | 0.210 | 0.173 | 0.153 |
| Plant life | 10–20 yr | 0.117 | 0.096 | 0.085 |
| Natural gas price | ×0.5–1.5 | 0.043 | 0.032 | 0.031 |
| Catalyst lifetime | 1–5 yr | 0.0012 | 0.0010 | 0.0009 |
| Hydrogen price | 1.0–3.34 | 0.000 | 0.000 | 0.000 |

## Findings
- **Feedstock cost dominates**, then discount rate ≈ CAPEX, then plant life. Same ranking all three cases.
- **Catalyst lifetime is a live but tiny lever**: shorter life → higher MSP (correct direction), swing ~0.001 $/kg. Catalyst is ~0.05 % of OPEX.
- **Hydrogen price has ZERO effect** — the published cash-flow model has **no H₂ purchase cost** (H₂ is produced internally by SMR of light gases; `C_feed` = biomass + NG only). The reviewer's H₂-price sensitivity cannot be answered without first adding an H₂ cost/credit term to the model. Flag for decision.

## Files
- `sa_oat_ijk.csv`, `sa_tornado_ijk.csv` — data
- `tornado_ijk.png`, `spider_ijk.png` — plots
- `sa_model_ijk.py` (model, ported from `cash_flow_SA.m`) + `run_sa_ijk.py` (sweeps) + `plot_sa_ijk.py` (plots). Base primitives from `SAF_Sensitivity_Analysis_260403_non_CC_all.xlsx`.
- Run: `PYTHONIOENCODING=utf-8 <saf_hdo>/python.exe run_sa_ijk.py && ... plot_sa_ijk.py`
