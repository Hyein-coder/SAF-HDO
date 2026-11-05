# Distribution-model comparison: HDO SimDist carbon-number data (11 experiments)

Script: `distfit/fit_dist.py` (`--selftest` runs the checks). Outputs:
- `distfit/results/{metrics,params,ranking,pooled}.csv` and `run_log.txt`
- `figures/distfit_{grid,rmse,rmse_heatmap}.{png,svg}`

## Data and setup
- **Source:** `HDO_exp.xlsx` Sheet1, Carbon 6–38 × cases a–k. Values are integer wt%, already converted from boiling point to carbon number by SimDist (n-paraffin basis). Each column sums to 100.
- **Variables:** the random variable is carbon number n; the frequency is w_n = wt%/100, on a **mass basis**.
- **Models as pmfs:** every model is a pmf on n = 6..38. Continuous models are binned as F(n+½) − F(n−½) and renormalized over [5.5, 38.5].
- **Fit:** minimize the cross-entropy −Σ w_n ln p_n (equivalently KL, the multinomial MLE on wt%) with bounded Nelder-Mead, 5 starts.
- **Support start.** Cases b–f report 0 wt% at C6.
  - The 3-parameter models fit a location θ, capped at (first reported carbon + ½).
  - nbinom and asf_shift start at the first reported carbon.
- **Comparison is on fit accuracy only, with no parameter penalty.** Metrics:
  - RMSE in wt%, over the first to the last reported carbon. This is the primary metric.
  - KL divergence, which is what the fit minimizes.
  - Maximum absolute error.
  - Error in the jet cut (C8–C16) and in the heavy fraction (C17+).
- **Rounding floor:** integer wt% limits RMSE to about 0.29 wt%.

## Models
| model | params | notes |
|---|---|---|
| normal | 2 | symmetric baseline |
| lognorm / lognorm3 | 2 / 3 | 3-parameter version adds location θ |
| gamma | 2 | Schulz–Zimm on mass basis |
| weibull3 | 3 | k, θ, λ; the form used in the 241217 slides |
| asf | 1 | textbook Anderson–Schulz–Flory, mass basis, W_n ∝ n αⁿ (chains counted from C1) |
| asf_shift | 1 | ASF counted from a seed at the first reported carbon (= nbinom with r = 2) |
| nbinom | 2 | negative binomial from the first reported carbon (r, q) |
| gengamma | 3 | Prentice form (μ, σ, Q). Nests lognormal, gamma and Weibull (all with loc = 0) |
| mix2ln | 5 | two-lognormal mixture, "cracked + uncracked" populations |

## Ranking by accuracy
Wins count the cases in which a model has the lowest RMSE, or the lowest KL.

| model | params | wins RMSE | wins KL | mean RMSE (wt%) | worst RMSE | mean KL | mean max \|err\| | mean \|Δ jet C8–16\| | mean \|Δ C17+\| |
|---|---|---|---|---|---|---|---|---|---|
| mix2ln | 5 | **7** | 5 | **0.78** | **1.31** | 0.033 | **1.83** | **0.75** | **0.65** |
| **weibull3** | 3 | 4 | **6** | 0.81 | 1.38 | **0.030** | 2.06 | 1.65 | 1.43 |
| lognorm3 | 3 | 0 | 0 | 0.92 | 1.35 | 0.042 | 2.18 | 2.75 | 2.26 |
| nbinom | 2 | 0 | 0 | 0.94 | 1.44 | 0.039 | 2.20 | 2.20 | 1.76 |
| gengamma | 3 | 0 | 0 | 0.96 | 1.33 | 0.044 | 2.33 | 1.88 | 1.63 |
| lognorm | 2 | 0 | 0 | 1.03 | 1.52 | 0.054 | 2.46 | 1.96 | 1.91 |
| gamma | 2 | 0 | 0 | 1.07 | 1.76 | 0.057 | 2.80 | 1.95 | 1.54 |
| asf_shift | 1 | 0 | 0 | 1.17 | 2.18 | 0.051 | 2.85 | 3.94 | 2.28 |
| normal | 2 | 0 | 0 | 1.23 | 2.20 | 0.071 | 3.87 | 3.98 | 1.38 |
| asf (from C1) | 1 | 0 | 0 | 1.94 | 3.78 | 0.151 | 7.63 | 11.96 | 3.61 |

RMSE (wt%) by case for the leading models:

| model | a | b | c | d | e | f | g | h | i | j | k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull3 | **0.92** | **1.09** | **1.10** | 0.62 | 0.89 | 1.37 | **0.43** | 0.60 | **0.42** | 0.72 | 0.79 |
| mix2ln | 0.93 | 1.31 | 1.28 | **0.59** | **0.69** | **1.24** | 0.49 | **0.49** | **0.42** | **0.62** | **0.48** |
| nbinom | 1.06 | 1.19 | 1.12 | 0.72 | 0.92 | 1.44 | 0.56 | 0.84 | 0.45 | 0.91 | 1.11 |
| lognorm3 | 1.04 | 1.16 | 1.35 | 0.70 | 1.03 | 1.35 | 0.62 | 0.71 | 0.46 | 0.84 | 0.90 |
| asf_shift | 1.98 | 1.49 | 1.37 | 0.76 | 1.05 | 2.17 | 0.64 | 0.92 | 0.44 | 0.96 | 1.03 |

**Pooled fits** share one shape across all 11 cases and fit the rest per case. Sharing the shape always costs accuracy:

| model | shared | mean RMSE, individual → pooled | mean KL, individual → pooled |
|---|---|---|---|
| weibull3 | k = 1.49, θ = 6.11 | 0.81 → 1.11 | 0.030 → 0.051 |
| nbinom | r = 2.33 | 0.94 → 1.11 | 0.039 → 0.049 |
| lognorm3 | s = 0.62, θ = 4.40 | 0.92 → 1.39 | 0.042 → 0.086 |

## Findings
1. **Weibull3 is the most accurate single-population model.**
   - It has the best mean KL of all models, including the 5-parameter mixture, and is a close second in mean RMSE (0.81 vs 0.78).
   - It ranks first or second in RMSE in 10 of 11 cases. The exception is f, where models 1–6 all fall within 1.24–1.37 wt%.
   - Its location θ ≈ 6–7 sits at the first reported carbon, and its shape k ≈ 1.2–2.0 (> 1) means the chance of a chain stopping rises with length. This matches the Weibull used in the 241217 slides.
2. **The two-lognormal mixture is marginally more accurate but poorly determined.**
   - It has the lowest RMSE in 7 cases and the best cut-fraction accuracy (jet 0.75, C17+ 0.65 wt%).
   - It needs 5 parameters, and 4 of 11 fits have weak parameters (the mixing fraction f is poorly determined), so its two "populations" are not reliably identified.
   - Its edge over Weibull3 is mostly in the heavy-tail and flat cases (d, e, h, j, k).
3. **Location matters more than the family.** The models with a fitted start, weibull3 and lognorm3, beat their loc-free relatives: gengamma, lognorm and gamma.
4. **Single-shape forms are less accurate.** nbinom, asf_shift and normal follow these, and the textbook ASF is far worst, with a jet-cut error of 12 wt%.
5. **A shared shape costs accuracy.** For weibull3, sharing k and θ across all runs raises mean RMSE from 0.81 to 1.11 wt%, so each run needs its own shape for an accurate fit.
6. **No model reaches the 0.29 wt% rounding floor.** The best mean is about 0.8. The remaining error is bin-to-bin scatter (e.g. d, e and j alternate between 4 and 6 wt%) that a smooth pmf cannot follow, likely from the SimDist n-paraffin binning.

## Recommendation
- **Use Weibull3, fitted per case,** as the carbon-number distribution model. It has the best KL, second-best RMSE by 0.04 wt%, and well-determined parameters. It also keeps continuity with the earlier slides.
- **Use mix2ln only if exact jet or heavy cut fractions matter more** than having stable parameters.

## Caveats
- **Accuracy-only ranking favors more parameters.** That favors mix2ln (5 parameters); Weibull3 matching it with 3 is the stronger result.
- **Fit objective vs ranking metric:** the fit minimizes KL, while ranking uses RMSE. A least-squares fit would lower each model's RMSE slightly but should not change the order much, because the two leaders are separated by a small, consistent margin.
- **Data limits:** the data are rounded to integer wt% and contain no C≤5 light ends. θ ≈ 6–7 partly reflects this detection edge.
- **Column mapping:** the mapping of columns a–k to Table 1 entries (A1–C2) is unconfirmed; see `ryield/PLAN_RYield_conversion.md`. It is needed before regressing Weibull parameters on T or catalyst.
- **Flags:**
  - `converged` (the Nelder-Mead stop flag) is weak.
  - Use `at_bound`, `unidentifiable` and `weak` instead.
  - `weak` leaves out the location-type parameters θ, μ and Q.

## Independent review
The code was reviewed by a separate Claude agent running `/code-review` (high, high, then medium), with three passes in total.

**Pass 1** found four HIGH issues, all fixed:
- The original ASF was identical to NB(r=2). It is now split into the textbook `asf` plus an explicit `asf_shift`.
- The θ cap was arbitrary. It is now set per case.
- gengamma could not reach its lognormal limit. It now uses the Prentice form.
- The pooled lognorm3 fit was stuck in a local minimum. It now uses multi-candidate starts.

**Pass 2** found medium issues, all fixed:
- A skipped fit crashed later stages. It now raises an error.
- The identifiability flags didn't work. The Hessian check now drops at-bound parameters, and a `weak` column was added.
- The gengamma switch was moved to |Q| < 1e-5, and RMSE now covers the reported range only.
- nbinom had to start at C6 while the θ models could start at C7. It now starts at the first reported carbon. This one changed the ranking.

**Pass 3** found no bugs. Its point that n0 is not counted is handled by the n0-counted columns. The (nbinom, asf_shift) pair was added to the nesting check.

After the user asked to compare on fit accuracy only, AICc, BIC and the pseudo-N were removed from the code. The ranking is now by RMSE and KL.
