# Literature survey: fitting final product distributions and modeling HDO from endpoint data

Date: 2026-11-17. Review type: focused scoping review.

## Main finding

The most useful literature for our current data concerns **mixture characterization, composition-surrogate fitting, and experiment-calibrated process models**. Reaction kinetics is a separate modeling objective. A distribution fitted to the final product can be useful without claiming to explain its development during reaction.

This review reframes the methodological recommendations in [the earlier recap](261117_literature_survey_recap.md). It retains the saved fitting results as project evidence, but does not use the Weibull–hydrocracking analogy to interpret fitted parameters as identified reaction rates or severity.

The core evidence comprises 12 journal/conference papers and one process-design report. Access depth varies and is recorded in [the source and search log](261117_endpoint_composition_sources.md). This is a targeted scoping review, not an exhaustive systematic review or a novelty certification.

## 1. Research question and scope

**Given measured final HDO product distributions, how should we represent composition, choose and assess a fitted distribution, and construct an Aspen model suitable for TEA/LCA?**

The working dataset is described in [distfit/REPORT.md](distfit/REPORT.md): 11 experiments a–k, C6–C38 bins, integer mass percentages, normalized to 100% per column. That record describes the bins as SimDist-derived on an n-paraffin basis. The original analytical method and conversion procedure were not independently established in this review; conclusions about equivalent carbon number are conditional on that description.

Included: empirical distribution fitting, boiling-range and molecular-weight characterization, pseudocomponent construction, measured-property surrogate fitting, endpoint product characterization, and process models using specified yields or conversions.

Excluded from the core evidence: time-evolution kinetics, microkinetics, fragmentation mechanisms, catalyst-development studies without relevant endpoint characterization, and fatty-acid/HEFA models presented as direct equivalents of pyrolysis-oil HDO. Kinetic papers are retained only where a specific endpoint-model implementation is useful.

## 2. Literature most relevant to the present task

### A. Distribution fitting and mixture characterization

| Source | Data and representation | What it supports for our work | Limit |
|---|---|---|---|
| **Whitson (1983)** — [Characterizing Hydrocarbon Plus Fractions](https://doi.org/10.2118/12233-PA) | Fits a three-parameter gamma function to petroleum-fraction weight/molar distributions; constructs single-carbon-number groups. | A probability distribution can characterize an incompletely specified mixture without being a reaction model. Provides a strong precedent for a fitted location parameter and careful mass/mole accounting. | Petroleum plus fractions; does not establish gamma or Weibull superiority for our HDO samples. |
| **Cotterman, Bender & Prausnitz (1985)** — [Continuous thermodynamics for chemical process design](https://doi.org/10.1021/i200028a033) | Describes complex mixtures by distributions over a property such as molecular weight or normal boiling point. | Connects a statistical mixture description with phase-equilibrium calculations. | Continuous thermodynamics describes mixture properties; it does not imply continuous reaction kinetics. |
| **Riazi (1997)** — [A Continuous Model for C7+ Fraction Characterization of Petroleum Fluids](https://doi.org/10.1021/ie970260y) | Uses a distribution model with bulk-property information; evaluates property distributions and fraction splitting/lumping. | A template for turning incomplete characterization into simulator-ready fractions while retaining additional property constraints. | Needs more than a normalized carbon-number histogram; calibrated for petroleum fluids. |
| **León, Riaño & Laverde (2016)** — [Prediction and extension of curves of distillation of vacuum residue using probability functions](https://doi.org/10.1088/1742-6596/687/1/012093) | Compares Weibull extreme, Weibull, Kumaraswamy and Riazi functions for measured distillation curves. Uses 50 residues for fitting and five for validation, with RSS and AIC/BIC assessment. | Direct precedent for selecting a probability function by endpoint-curve accuracy and validation. | The preferred Weibull-extreme form is not automatically our three-parameter Weibull. Its substrate, coordinate and fitted equations differ. |

**Synthesis:** These studies justify the *type of task* we are doing: representing a complex mixture by a compact distribution. The choice of the best family for our samples must come from our own comparison. A good fit in boiling temperature or molecular weight is not automatically a good fit in carbon number, because transforming the coordinate changes the distribution.

### B. Fitting representative compositions to measured properties

| Source | Calibration target or representation | What it supports for our work | Limit |
|---|---|---|---|
| **Huber, Lemmon & Bruno (2010)** — [Surrogate Mixture Models for the Thermophysical Properties of Aviation Fuel Jet-A](https://doi.org/10.1021/ef100208c) | Constructs small representative mixtures using distillation curves and several measured thermophysical properties. | A fitted representative composition can be useful even when it is not a complete molecular assay. Multiple property targets constrain the representation. | Jet-A property surrogates; does not identify HDO pathways or hydrogen demand. |
| **Lozano, Pedersen & Rosendahl (2019)** — [Modeling of thermochemically liquefied biomass products and heat of formation for process energy assessment](https://doi.org/10.1016/j.apenergy.2019.113654) | Optimizes representative compounds against measured bulk properties for pyrolysis oil and HTL biocrude; implements the representation in Aspen. | A particularly relevant precedent for composition fitting when exact stoichiometry and kinetics are unknown. Also links surrogate-property choices with energy assessment. | Fits bulk-property representations rather than our exact carbon-number endpoint target. |
| **Iwube, Li & Brightman (2025)** — [Process simulation of the integration of molecular distillation with fast pyrolysis of biomass for sustainable fuel production](https://doi.org/10.1039/d4se01077a) | Section 2.1 uses an Excel solver with normalized feedstock, oxygenate, ultimate-analysis and product-yield targets; assigned component yields enter Aspen RYield. | Concrete example of solving a constrained composition problem and supplying its result to a process simulator. | Pyrolysis/separation example, not an HDO reactor calibration. Its separation claims should not be read as proof of chemical deoxygenation. |

**Synthesis:** This is the closest methodological family to our fitting workflow. The optimizer selects an effective representation within an assumed component set and model structure. The optimization algorithm itself does not determine whether the resulting parameters are identifiable or physically unique. A fitted Aspen representation should be described by its observations, constraints and intended use.

### C. Endpoint HDO characterization and flowsheet representation

| Source | Relevant evidence | What it supports for our work | Limit |
|---|---|---|---|
| **Christensen et al. (2011)** — [Analysis of Oxygenated Compounds in Hydrotreated Biomass Fast Pyrolysis Oil Distillate Fractions](https://doi.org/10.1021/ef201357h) | Characterizes boiling-range fractions from three hydrotreated oils using elemental and chemical analyses. | Endpoint fuel-cut distributions and chemical composition provide complementary information. | Characterization study; does not validate our fitted reaction coefficients. |
| **Olarte et al. (2017)** — [Characterization of upgraded fast pyrolysis oak oil distillate fractions from sulfided and non-sulfided catalytic hydrotreating](https://doi.org/10.1016/j.fuel.2017.03.051) | Measures five fractions from two upgraded oils and distinguishes chemical features across the fractions. | A template for validating more than the boiling/carbon-number profile when measurements exist. | Neither a universal product distribution nor a transferable kinetic model. |
| **Ma et al. (2019)** — [Process Simulation and Economic Evaluation of Bio-Oil Two-Stage Hydrogenation Production](https://doi.org/10.3390/app9040693) | Section 2.2 places mild-stage kinetics in RCSTR and the deep stage in RStoic. | Specific precedent for a stoichiometric HDO block within an integrated economic model. | A hybrid model: citing it does not justify calling the whole process either kinetic or fixed-yield. |
| **Tuppurainen, Nissilä & Tanskanen (2026 issue; online 2025)** — [Process simulation and techno-economic analysis of fast pyrolysis bio-oil hydrodeoxygenation using a slurry catalyst reactor](https://doi.org/10.1016/j.ces.2025.122567) | Uses representative functional-group components for two-stage HDO in Aspen and reports product distillation comparisons. | Closely matched application context: HDO, surrogate components, fuel fractions and TEA. | Full reactor-method details were not reliably retrieved; no RYield/RStoic block assignment is asserted here. |
| **Jones et al. (2013), PNNL-23053 / NREL/TP-5100-61178** — [Fast Pyrolysis and Hydrotreating Bio-Oil Pathway](https://www.pnnl.gov/publications/process-design-and-economics-conversion-lignocellulosic-biomass-hydrocarbon-fuels-fast) | Process-design and economic assessment based on conversion targets. | Benchmark for connecting a stated process-performance basis with economics. | Design targets must be distinguished from validated experimental prediction; the reactor implementation was not reaudited in this review. |

**Synthesis:** Endpoint-calibrated process modeling is a useful and defensible objective. Its validity depends on which outputs are actually measured, which are assumed, and whether the process model preserves the required balances. The literature does not make a fitted outlet distribution sufficient evidence for all other flowsheet outputs.

## 3. The measurement coordinate needs explicit treatment

**A SimDist-derived carbon-number profile may be an equivalent boiling-range representation rather than a molecular carbon-number assay.** The distinction should be resolved before manuscript wording is finalized.

The [ASTM D2887 method description](https://store.astm.org/d2887-24.html) identifies its output as a boiling-range distribution. An [instrument-provider implementation](https://scioninstruments.com/wp-content/uploads/2020/12/0008_ASTMD2887_SimDist.1.pdf) describes the use of n-paraffins to calibrate retention time against boiling point. These are analytical-method references, not evidence that our samples were measured by that particular ASTM method.

Our inference is that, if the bins were constructed from such a calibration, labeling a bin C12 does not establish that every molecule contributing to it contains exactly 12 carbon atoms. Compound classes can have different boiling behavior. A simulated true-carbon-number histogram and a measured equivalent-carbon-number histogram may therefore be different observables.

**Harries, McDonald & Bruno (2017)** — [Measuring the distillation curves of non-homogeneous fluids](https://doi.org/10.1016/j.fuel.2017.04.066) — additionally shows that water/organic phase separation requires care in pyrolysis-oil distillation analysis. This matters if upstream or incompletely upgraded oils are characterized; it is not evidence that our final products are multiphase.

For the current dataset, confirm the instrument/method, carbon-number conversion, mass basis, handling of light ends and residue, and whether normalization covers the entire sample or only the reported window. If raw boiling-point data are available, comparing simulated and measured boiling distributions would reduce reliance on the carbon-number conversion.

## 4. What our current endpoint data can support

The following is an assessment of our inverse problem, not a result claimed by any single cited paper.

| Quantity or claim | Supported by a normalized final distribution alone? | Additional information or assumption |
|---|---|---|
| Relative composition across the reported bins | Yes, within measurement and fitting error. | Match the measurement coordinate and normalization. |
| Empirical Weibull or alternative distribution parameters | Yes, as descriptors of that observed distribution. | Check bounds and stability; avoid kinetic labels. |
| Relative mass in a selected reported range | Yes, on the same basis as the histogram. | C8–C16 is a chosen cut metric, not certification of aviation-fuel suitability. |
| Absolute SAF production rate | No. | Total organic-liquid yield, feed throughput and recovery basis. |
| Molecular class composition and fuel properties | Not uniquely. | Speciation, PIONA/GC analysis, density, elemental or other property data. |
| Hydrogen consumption and carbon loss | Not uniquely. | Feed/product elemental balances and gas, aqueous, coke and hydrogen measurements or stated assumptions. |
| Stage-specific conversions or reaction pathways | Generally not uniquely in this multistage model. | Intermediate samples or independently justified constraints. |
| Reaction rates and responses to temperature/pressure/time | Not established by the final histogram fit. | Additional condition/time information and an identifiable model. |

For example, products with similar size distributions can have different aromatic contents or residual oxygen. Their hydrogen requirements and thermophysical properties can differ. Fitting the size histogram does not resolve those differences.

The [case-k RStoic-to-RYield report](ryield/REPORT_RYield_k.md) is relevant internal evidence: it records matching the converged base-point simulation with specified stage yields. That is equivalence to the reference simulation at that point, not new validation against experimental stage compositions or operating-condition changes.

## 5. Implications for our distribution comparison

The [saved project comparison](distfit/REPORT.md) found Weibull3 to have the lowest RMSE in 10 of 11 cases, with mean RMSE approximately 0.81 wt%. That result remains useful. No fitting was rerun for this review.

Recommended interpretation:

1. **Keep the existing empirical conclusion.** Weibull3 performed best among the distributions and fitting settings compared for these data. Avoid a universal claim about all HDO products.
2. **Use descriptive parameter names.** Shape, location and scale summarize the observed profile. The current observations do not identify a cracking-rate exponent, an unreactive carbon threshold or a reaction-severity parameter.
3. **Keep the current accuracy-only ranking explicit.** Literature use of AIC/BIC does not require changing the user's chosen comparison. If inferential statistics are later introduced, the data/error model must be justified. Rounded mass percentages are not counts of independently sampled molecules, and their sum of 100 is not a statistical sample size.
4. **Respect the reported support.** Window renormalization describes composition conditional on that window. It cannot recover unmeasured gas/light products or heavy residue. Integer reporting also does not establish the experimental uncertainty.
5. **Separate numerical fit from parameter interpretation.** Case k's boundary-driven Weibull fit is a reason to report limited interpretability even if its curve error is acceptable. A flexible family can fit an endpoint through poorly constrained parameters.

Whitson's three-parameter gamma is also a reminder that the saved two-parameter gamma comparison does not establish that every gamma formulation is inferior. This is a comparison-scope observation, not a proposal to expand or rerun the fitting in this task.

## 6. How to describe our HDO model

Appropriate framing is **an experiment-calibrated stoichiometric process model** or **an endpoint-calibrated product-composition model**, depending on the actual Aspen implementation.

For RStoic, explain which reaction set is assumed and which conversions or coefficients were adjusted. For RYield, explain which yields are specified, their basis, how hydrogen/unreacted species are handled, and how elemental closure is checked. A total-mass normalization alone does not independently demonstrate elemental consistency.

The minimum complete account should identify:

- the measured outlet target and its analytical basis;
- the selected representative components and assumed reactions;
- the fitted parameters and physical constraints;
- which absolute yields, hydrogen demands and phase compositions are independently known;
- distribution-fit errors and any separate validation;
- the base operating conditions and the supported use of the model in TEA/LCA.

Final-product fitting alone should not be described as independent validation of each HDO stage. Likewise, varying an economic input at a fixed Aspen material balance is a valid scenario calculation, but does not establish a chemical response to changed reactor conditions.

## 7. Suggested manuscript framing

The following is proposed wording to adapt after confirming the analytical basis; it is not a quotation from the literature:

> The final upgraded-oil distribution was represented using empirical probability functions and compared with the measured product profile. Distribution parameters were treated as descriptors of the observed composition. An assumed stoichiometric reaction framework was calibrated to reproduce the endpoint distribution for process simulation. Reaction kinetics and the uniqueness of the fitted reaction pathway were not established by this calibration. Absolute product yields, hydrogen consumption and other process quantities were taken from independently specified data or assumptions, as identified separately.

If the analytical conversion is confirmed, replace “product profile” with “n-paraffin-equivalent boiling-range distribution expressed in carbon-number bins.” If the data instead come from molecularly resolved analysis, use the appropriate measured carbon-number terminology.

## 8. Reading priorities and remaining gaps

For immediate methodological support, prioritize **Whitson (distribution characterization), León et al. (endpoint fitting/comparison), Lozano et al. (bio-oil composition fitting), Iwube et al. (constrained yields into Aspen), and Tuppurainen et al. (closely matched HDO/TEA context)**. Christensen and Olarte are the most useful sources here for explaining the limits of a size/boiling profile as a complete characterization.

The search did not establish an exact published precedent for our particular BO-to-Aspen carbon-bin fitting workflow. That absence in a focused search is not evidence of novelty. Full-text checking of the most relevant partially accessible papers remains necessary before using detailed methodological claims in a submission.

Outstanding project questions are the analytical conversion, whole-product versus window normalization, absolute liquid yield, independently constrained hydrogen demand, and the experimental basis for intermediate-stage compositions. Resolving those questions will improve the model's defensibility more directly than adding a kinetic interpretation to the fitted distribution.
