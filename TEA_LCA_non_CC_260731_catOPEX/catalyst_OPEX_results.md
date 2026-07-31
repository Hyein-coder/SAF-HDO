# Catalyst OPEX included in C_util — MSP impact (cases i, j, k, non_CC)

Change: `C_util(end+1) = processData.OperatingCost.Catalyst` in all 3 cash_flow files.
Catalyst OPEX = 59,707.35 / 3 (lifetime) = **19,902.45 $/yr** now enters COM.

MSP_new = published MSP + validated delta (Python port reproduces published MSP within ~1.4%;
the delta is a derivative and is offset-insensitive; analytic ΔMSP=ΔC·1.1876/(mSAF·Top) matches numeric).

| case | MSP_pub [$/kg] | ΔMSP [$/kg] | MSP_new [$/kg] | %chg | NPV_pub@1.1321 [$] | ΔNPV [$] |
|---|---|---|---|---|---|---|
| j | 1.084655 | +0.000407 | 1.085062 | +0.0375% | 6,035,315 | -52,370 |
| k | 0.949573 | +0.000356 | 0.949928 | +0.0375% | 26,565,885 | -52,370 |
| i | 1.406130 | +0.000493 | 1.406623 | +0.0351% | -28,778,481 | -52,370 |

**Conclusion:** catalyst OPEX raises MSP by ~0.036–0.038% (~0.0004 $/kg) and lowers NPV by ~50k$. Negligible.
