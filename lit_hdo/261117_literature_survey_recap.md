# Literature survey recap: distribution fitting and HDO reaction modeling

Recap saved: 2026-11-17. Revised 2026-11-18 to exclude MDPI publications and state study applications. Literature survey dates: 2026-11-04–05, with subsequent distribution-fitting results.

The main conclusion was: **Weibull3 worked best for describing our measured carbon-number distributions; predicting their response to HDO conditions still required a calibrated reaction model.**

The records are [the literature survey](REPORT.md) and [the fitting report](../distfit/REPORT.md).

## Distribution fitting

We explored two directions. First, we looked for physical interpretations of carbon-number distributions:

| Literature | Goal/application in the study | What it contributed |
|---|---|---|
| **Flory (1936); Anderson et al. (1951)** | Describe size distributions from polymer chain growth and Fischer–Tropsch chain growth/termination. | Chain-growth/termination foundations for ASF distributions; useful as a baseline, although developed for polymerization and Fischer–Tropsch chemistry. |
| **Brown & Wohletz (1995)** | Derive particle-size distributions from fragmentation principles. | A physical derivation of Weibull distributions for fragmentation. This motivated a fragmentation interpretation, but does not establish a molecular HDO mechanism. [Paper](https://doi.org/10.1063/1.360073) |
| **Laxminarasimhan et al. (1996)** | Simulate hydrocracking with a continuous mixture distribution. | Continuous hydrocracking model using boiling point, size-dependent reactivity, and a product-formation kernel. This was the main bridge between a fitted distribution and reaction kinetics. [Paper](https://doi.org/10.1002/aic.690420925) |
| **Elizalde & Ancheyta (2009–2014)** | Model heavy-oil hydrocracking responses to operating conditions and deactivation. | Extensions involving operating conditions, numerical solution, and deactivation. The 2010 paper supplied the template for making distribution behavior depend on temperature and pressure. [2010 paper](https://www.sciencedirect.com/science/article/pii/S0926860X10003236) |

Second, we compared **nine distributions against all 11 experiments, a–k**, using C6–C38 mass fractions from `HDO_exp.xlsx`. Continuous distributions were integrated into carbon-number bins and fitted by minimizing KL divergence.

| Model | Mean RMSE, wt% | Cases with lowest RMSE |
|---|---:|---:|
| **Three-parameter Weibull** | **0.81** | **10/11** |
| Three-parameter lognormal | 0.92 | 0 |
| Negative binomial | 0.94 | 0 |
| Generalized gamma | 0.96 | 0 |
| Conventional ASF | 1.94 | 0 |

The recorded decision was to **retain Weibull3 with parameters fitted separately for each experiment**. Sharing its shape and location across experiments increased mean RMSE from 0.81 to 1.11 wt%. This comparison assessed fitting accuracy without a parameter-count penalty.

We also explored interpreting Weibull shape, location, and scale as size selectivity, a lower-size threshold, and severity. **That mapping remains a formal analogy.** It omits product formation and depends on assumptions about the feed and boiling-point/carbon-number relationship. Case **k** especially gave boundary-driven parameters without a useful physical interpretation. [Fitting discussion](../distfit/REPORT.md)

## HDO reaction modeling

The survey covered seven model families: lumped kinetics, model-compound kinetics/microkinetics, continuous lumping, molecular reconstruction and reaction networks, equilibrium models, ML surrogates, and empirical severity correlations.

The most relevant studies for our process were:

- **Zhang (2009):** an early four-lump bio-oil hydrotreating model, identified in the saved survey as a compact kinetic starting point.
- **Cordero-Lanzac et al. (2020):** raw-bio-oil HDO kinetics incorporating transient behavior and catalyst deactivation—the strongest candidate for reaction-network structure. Its calibration remains specific to its feed/catalyst system. [Paper](https://www.sciencedirect.com/science/article/pii/S1385894720306707)

The earlier search **did not identify a validated model spanning major catalyst families while predicting whole-pyrolysis-oil carbon-number distributions**. That is a finding of the survey’s coverage, rather than proof that none exists.

## Proposed next step and unresolved prerequisites

The proposed next step was a **condition-dependent yield/severity model**: connect our fitted conversions or distribution parameters to temperature, hydrogen partial pressure, and residence time, retaining catalyst-specific calibration. Validation would jointly use carbon-number distributions, hydrogen uptake, oxygen removal, and gas/liquid/aqueous yields.

The main unresolved prerequisite was suitable multi-condition data; the experiment-label-to-condition mapping also remained unconfirmed.

## Coverage limits

The fits were not rerun and an exhaustive new literature search was not performed for this recap. The key limitation remains that a good distribution fit alone does not validate predictive HDO kinetics.
