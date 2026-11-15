# Distribution-model comparison: HDO SimDist carbon-number data (11 experiments)

Script: `distfit/fit_dist.py` (`--selftest` runs the checks). Outputs:
- `distfit/results/{metrics,params,ranking,pooled,weibull_laxmi}.csv` and `run_log.txt`
- `figures/distfit_{grid,rmse,rmse_heatmap,laxmi}.{png,svg}`

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
- **Excluded:** the two-lognormal mixture was dropped at the user's request.

## Models
| model | params | notes |
|---|---|---|
| normal | 2 | symmetric baseline |
| lognorm / lognorm3 | 2 / 3 | 3-parameter version adds location θ |
| gamma | 2 | Schulz–Zimm on mass basis |
| **weibull3** | 3 | shape k, location θ, scale λ; the form used in the 241217 slides |
| asf | 1 | textbook Anderson–Schulz–Flory, mass basis, W_n ∝ n αⁿ (chains counted from C1) |
| asf_shift | 1 | ASF counted from a seed at the first reported carbon (= nbinom with r = 2) |
| nbinom | 2 | negative binomial from the first reported carbon (r, q) |
| gengamma | 3 | Prentice form (μ, σ, Q). Nests lognormal, gamma and Weibull (all with loc = 0) |

## Ranking by accuracy
Wins count the cases in which a model has the lowest RMSE, or the lowest KL.

| model | params | wins RMSE | wins KL | mean RMSE (wt%) | worst RMSE | mean KL | mean max \|err\| | mean \|Δ jet C8–16\| | mean \|Δ C17+\| |
|---|---|---|---|---|---|---|---|---|---|
| **weibull3** | 3 | **10** | **10** | **0.81** | 1.37 | **0.030** | **2.06** | **1.65** | 1.43 |
| lognorm3 | 3 | 0 | 0 | 0.92 | 1.35 | 0.042 | 2.18 | 2.75 | 2.26 |
| nbinom | 2 | 0 | 0 | 0.94 | 1.44 | 0.039 | 2.19 | 2.20 | 1.76 |
| gengamma | 3 | 0 | 1 (k) | 0.96 | **1.33** | 0.044 | 2.33 | 1.88 | 1.63 |
| lognorm | 2 | 1 (f) | 0 | 1.03 | 1.52 | 0.054 | 2.46 | 1.96 | 1.91 |
| gamma | 2 | 0 | 0 | 1.07 | 1.76 | 0.057 | 2.80 | 1.95 | 1.54 |
| asf_shift | 1 | 0 | 0 | 1.17 | 2.17 | 0.051 | 2.85 | 3.94 | 2.28 |
| normal | 2 | 0 | 0 | 1.23 | 2.20 | 0.071 | 3.87 | 3.98 | **1.38** |
| asf (from C1) | 1 | 0 | 0 | 1.94 | 3.78 | 0.151 | 7.63 | 11.96 | 3.61 |

RMSE (wt%) by case for the leading models:

| model | a | b | c | d | e | f | g | h | i | j | k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| weibull3 | **0.92** | **1.09** | **1.10** | **0.62** | **0.89** | 1.37 | **0.43** | **0.60** | **0.42** | **0.72** | **0.79** |
| lognorm3 | 1.04 | 1.16 | 1.35 | 0.70 | 1.03 | 1.35 | 0.62 | 0.71 | 0.46 | 0.84 | 0.90 |
| nbinom | 1.06 | 1.19 | 1.12 | 0.72 | 0.92 | 1.44 | 0.56 | 0.84 | 0.45 | 0.91 | 1.11 |
| lognorm | 1.04 | 1.58 | 1.38 | 1.02 | 1.14 | **1.33** | 0.65 | 0.92 | 0.48 | 0.98 | 1.13 |

**Pooled fits** share one shape across all 11 cases and fit the rest per case. Sharing the shape costs accuracy:

| model | shared | mean RMSE, individual → pooled | mean KL, individual → pooled |
|---|---|---|---|
| weibull3 | k = 1.49, θ = 6.11 | 0.81 → 1.11 | 0.030 → 0.051 |
| nbinom | r = 2.33 | 0.94 → 1.11 | 0.039 → 0.049 |
| lognorm3 | s = 0.62, θ = 4.40 | 0.92 → 1.39 | 0.042 → 0.086 |

## Findings
1. **Weibull3 is the most accurate model.** It has the lowest RMSE in 10 of 11 cases and the lowest KL in 10 of 11, with a mean RMSE of 0.81 wt%. The two exceptions are near-ties:
   - In case f, lognorm has 1.33 wt% vs 1.37, and models 1–5 all fall within 1.33–1.44.
   - Case k is discussed below.
2. **Location matters more than the family.** The models with a fitted start, weibull3 and lognorm3, beat their loc-free relatives: gengamma, lognorm and gamma.
3. **Single-shape forms are less accurate.** nbinom, asf_shift and normal follow, and the textbook ASF is far worst, with a jet-cut error of 12 wt%.
4. **Each run needs its own shape.** Sharing k and θ across all runs raises Weibull3 mean RMSE from 0.81 to 1.11 wt%.
5. **No model reaches the 0.29 wt% rounding floor.** The best mean is about 0.8. The remaining error is bin-to-bin scatter (e.g. d, e and j alternate between 4 and 6 wt%) that a smooth pmf can't follow, likely from the SimDist n-paraffin binning.

## Interpreting the Weibull3 parameters with Laxminarasimhan et al. (1996)
Source: Laxminarasimhan, Verma & Ramachandran, "Continuous lumping model for simulation of hydrocracking", *AIChE J.* 42(9):2645–2653 (1996). The PDF is in the ObsidianVault at `KAN/VibeCoding/References/biomass/`.

### The paper's model
- **Characterization (Eq. 2):** the mixture is characterized by the normalized boiling point θ_TBP = (TBP − TBP_l)/(TBP_h − TBP_l), which runs from 0 at the lightest species to 1 at the heaviest.
- **Rate law (Eq. 4):** each species cracks by first-order kinetics with a rate constant that rises as a power of θ_TBP:

  k / k_max = θ_TBP^(1/α),   with k = 0 at θ_TBP = 0.

  The lightest species do not crack; heavier ones crack faster.
- **Mass balance (Eq. 8):** dc(k,t)/dt = −k·c(k,t) + ∫_k^k_max p(k,K)·K·c(K,t)·D(K) dK.
  - The first term is the loss of species k by cracking.
  - The integral is the formation of k from heavier species K. Its yield kernel p(k,K) is a skewed Gaussian (Eqs. 10–12) with parameters a₀, a₁ and δ.
  - D(k) = Nα/k_max^α · k^(α−1) is the species-type density (Eq. 6).
- **Fitted values in the paper (VGO hydrocracking):**
  - α = 1.35, k_max = 1.35 h⁻¹ for the Bennett–Bourne data.
  - α = 0.77, k_max = 0.88 h⁻¹ for El-Kardy at 390 °C.

### Mapping onto Weibull3
Assume first-order loss only and a normalized boiling point linear in carbon number, θ_TBP = (n − n_l)/(n_h − n_l). Then the fraction of species n surviving to residence time τ is

exp[−k_max·τ·((n − n_l)/(n_h − n_l))^(1/α)] = exp[−((n − θ)/λ)^k],

which is exactly the Weibull survival function. The parameters correspond one to one:

| Weibull3 | Laxminarasimhan | meaning |
|---|---|---|
| shape k | 1/α | how steeply the cracking rate rises with carbon number (k > 1 means faster than linear) |
| location θ | n_l (TBP_l) | lightest carbon, where the cracking rate goes to zero |
| scale λ | (n_h − n_l)·(k_max·τ)^(−α) | characteristic product size; smaller λ means higher severity k_max·τ |

`figures/distfit_laxmi.png` checks this power law directly. On log-log axes, −ln(mass fraction heavier than n) against (n − θ) should be a straight line of slope 1/α (the dashed line).
- **a–j:** the data follow the line over 1–2 decades.
- **Upward bend at the heavy end:** this comes from the C38 window truncation, and the solid line (the truncated fit) reproduces it.
- **Case k:** not a power law.

### Fitted values (`distfit/results/weibull_laxmi.csv`)
| case | 1/α = k | α | θ = n_l | λ | severity k_max·τ (n_h = C38.5) |
|---|---|---|---|---|---|
| a | 1.98 | 0.50 | 4.9 | 6.5 | 25.8 |
| f | 1.77 | 0.57 | 6.4 | 6.2 | 18.6 |
| c | 1.52 | 0.66 | 6.5 | 5.5 | 14.3 |
| b | 1.30 | 0.77 | 7.0 | 5.0 | 10.9 |
| g | 1.46 | 0.68 | 6.1 | 9.7 | 5.8 |
| d | 1.25 | 0.80 | 6.9 | 9.4 | 4.6 |
| e | 1.34 | 0.75 | 6.5 | 10.9 | 4.2 |
| j | 1.24 | 0.81 | 6.2 | 12.2 | 3.3 |
| h | 1.18 | 0.85 | 6.3 | 12.3 | 3.1 |
| i | 1.39 | 0.72 | 6.0 | 15.2 | 2.9 |
| k | 7.96 | 0.13 | −40 (bound) | 54.4 | not meaningful |

### What the parameters say
1. **α = 0.50–0.85 (cases a–j): the cracking rate rises faster than linearly with carbon number** (k ∝ θ_TBP^1.2–2.0), so heavier molecules are cracked disproportionately faster. This range sits around the paper's El-Kardy value (α = 0.77) and below its Bennett–Bourne value (1.35). The light runs a, f and c have the smallest α, meaning the strongest size-selectivity.
2. **θ ≈ 6–7 for b–j and 4.9 for a.** The zero-rate end of the rate law sits at the first reported carbon. In the paper's terms, C6–C7 is TBP_l, the lightest species, which no longer cracks. This is partly also the SimDist detection edge, since C≤5 is not reported, so θ isn't purely chemical.
3. **Severity k_max·τ splits the runs into two groups.**
   - High severity, light product: a, f, c, b (11–26).
   - Low severity, heavy tail: g, d, e, j, h, i (2.9–5.8).

   The absolute values depend on the assumed heaviest carbon n_h. The **ranking of a–j is identical for n_h = C30.5, C38.5 and C50.5**, so the ordering is robust. This index is the natural single number to regress on temperature, catalyst and residence time (Direction 2).
4. **Case k doesn't fit the Laxminarasimhan form.** Its distribution is flat and near-symmetric, and Weibull3 fits it only through the degenerate limit (θ at its −40 bound, k ≈ 8, with 21 % of the untruncated mass below C5.5). Its α and severity have no physical meaning.

### Caveats (important)
The mapping is a **formal analogy**, not a derivation.

Laxminarasimhan's exp(−kτ) is the fraction of one species that survives. The Weibull survival here is the *cumulative* product mass heavier than n. Equating the two requires all of the following:
- **(a) No formation term.** The integral in Eq. 8 has to be dropped, yet formation of lighter species from heavier ones is what fills the light end of a cracked product. The paper's α was fitted with the full Eq. 8, so the α comparison above is only order-of-magnitude.
- **(b) A specific feed shape.** The feed mass has to be uniform in rate constant k, i.e. c₀(n) ∝ (n − n_l)^(1/α − 1). That piles mass toward heavy species, which is not a pyrolysis-oil feed.
- **(c) k_max·τ ≫ 1.** In h, i and j it is 2.9–3.3, and 3.5–5.6 % of the untruncated Weibull mass would sit above C38 (`sf_above`).
- **(d) Boiling point linear in carbon number.** SimDist n-paraffin TBP is concave in n and depends on compound class.

To identify α, k_max, a₀, a₁ and δ properly, the full Eq. 8 has to be fitted to a feed (pyrolysis-oil SimDist) plus products at more than one residence time. Weibull3 collapses feed, kernel and time into three numbers.

Also note the paper describes VGO hydrocracking, whereas this is pyrolysis-oil HDO. HDO adds deoxygenation and possible oligomerization (k > K), which the paper's p(k,K) = 0 for k > K excludes.

## Recommendation
- **Use Weibull3, fitted per case,** as the carbon-number model: it has the best RMSE and KL in 10 of 11 cases.
- **Report its parameters in Laxminarasimhan terms:** α (rate–size exponent), n_l (lightest uncracked carbon), and severity k_max·τ, citing the analogy caveats above.
- **Treat case k separately.**

## Caveats (fit)
- **Fit objective vs ranking metric:** the fit minimizes KL, while ranking uses RMSE. The two agree on the winner in 9 of 11 cases.
- **Data limits:** the data are rounded to integer wt% and contain no C≤5 light ends.
- **Column mapping:** the mapping of columns a–k to Table 1 entries (A1–C2) is unconfirmed; see `ryield/PLAN_RYield_conversion.md`. It is needed before linking severity to T or catalyst.
- **Flags:**
  - `converged` (the Nelder-Mead stop flag) is weak.
  - Use `at_bound`, `unidentifiable` and `weak` instead.

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

**Pass 4** (medium) covered the mixture removal and the Laxminarasimhan mapping. It confirmed the algebra and found the removal clean. Three things changed as a result:
- The plot is now truncation-aware: the solid line is the window-renormalized fit, and the dashed line is the pure power law.
- The severity column was renamed `kmax_tau_C38` and documented as an untruncated extrapolation.
- The assumptions (a)–(d) above were written down.
