# HDO reaction modeling — literature survey (2026-11-04)

Source: 3 Codex (`codex exec`, web search, reasoning=medium) runs — raw outputs in `codex/out_*.md`, prompts in `codex/*.md`.
All 22 cited DOIs checked against Crossref (all resolve). Attribution errors found and corrected below.

## Bottom line

- **No catalyst-agnostic raw-bio-oil HDO model exists** that is validated across sulfided NiMo/CoMo, noble metals and Ni *and* predicts C5–C20+ distribution. All three Codex runs independently converged on this.
- The closest things to "catalyst-agnostic" are (i) Gibbs equilibrium (catalyst-free, but only an endpoint bound), (ii) severity/H2-vs-DOD correlations (catalyst omitted, not transferred), (iii) multi-catalyst ML with catalyst *descriptors* (catalyst-aware, model-compound only), (iv) DFT microkinetics across metals (compares catalysts, no transferable rate law).
- Process TEAs (PNNL/NREL design reports, most journal TEAs) use **fixed-yield RYield/RStoic** — same limitation as our case k. Only a few embed lumped kinetics in Aspen RCSTR (Ma 2019, Bagnato & Sanna 2019).
- So: our BO-fitted carbon-number stoichiometry is in the mainstream; the gap = make it condition-responsive.

## Model families (Codex A)

| Family | Representative | Catalyst handling | C-number output? | In Aspen? |
|---|---|---|---|---|
| Lumped raw-oil kinetics | Zhang 2009 (4 lumps); Cordero-Lanzac 2020 (7 lumps + deactivation, CoMo/AC) | in fitted k's | only if lumps defined by C# / bp | possible (RCSTR/RPlug); Ma 2019, Bagnato 2019 did |
| Model-compound kinetics / microkinetics | Xiao 2020 (guaiacol, Pt/C); Morteo-Flores 2023 (DFT, Co/Ni/Cu/Pd/Pt) | explicit (surface energetics) | no (compound-level) | no full flowsheet |
| Continuous / distributed lumping | none found validated for pyrolysis-oil HDO | — | would be native | gap |
| Molecular reconstruction + reaction rules + kMC | Alvarez-Majmutov 2025 (biocrude) | tuned rates + QSRR | derivable | no |
| Gibbs equilibrium | RGibbs | none (agnostic) | depends on species list | yes, but endpoint only |
| ML / surrogate / hybrid | Chen 2024 (guaiacol, literature dataset, catalyst textural features) | catalyst descriptors | no | external surrogate |
| Empirical severity / H2 vs DOD | Venderbosch 2010 scheme; PNNL design basis | omitted, feed-conditioned | no | trivial calculator |

## How TEAs represent HDO (Codex B)

| Source | Reactor | Predictive vs conditions? | Note |
|---|---|---|---|
| Jones 2013 PNNL-23053 | yields/conversions assumed | no | 5.8 wt% H2/dry oil; 68% C in HT oil |
| Dutta 2015 NREL/TP-5100-62455 | design yields | no | 91% hydroprocessing C efficiency |
| Ma 2019 Appl Sci | Aspen RCSTR, 9 pseudocomponents, coking deactivation | partly | 45% product yield |
| Bagnato & Sanna 2019 Catalysts | Aspen + lumped kinetics, boiling cuts incl. kerosene | partly | lump-to-jet-cut mapping template |
| Liu & Wang 2019 BBB | Aspen, experiment-anchored | unclear | jet 9 wt% / 21% C; MJSP $3.21/L |
| Okolie 2024 BioEnergy Res | Aspen → ML surrogate of **MSP** | no (econ only) | Aspen-to-ML TEA workflow |
| Farooq 2025 ECM | TEA meta-analysis | n/a | H2 consumption = top MJSP lever for pyrolysis-jet |
| Snowden-Swan 2020 (PNNL, DOI unverified) | experimental per-feedstock | no | catalyst 13–18% of MFSP |

Takeaway: published SA on H2 use / yield / WHSV is almost always done on fixed yields — same caveat applies to our SA (see memory `ryield-k-conversion`).

## Catalyst-independent insights (Codex C)

- HDO (O→H2O) keeps carbon but costs H2; DCO/DCO2 save H2 but lose C. Trade-off is feed/pathway dependent, not one curve.
- Reactivity order (carbonyls/acids/alcohols easy; phenolics, furanic ethers refractory) holds directionally; catalyst/support acidity can reorder.
- Low-T stabilization before deep HDO suppresses repolymerization/coking (Venderbosch 2010; de Miguel Mercader 2011 AIChE J) — justifies our 3-stage layout.
- Higher T favors cracking/DCO; higher pH2 favors hydrogenation/HDO. Directional, not quantitative laws.

## Gaps

1. Kinetic transferability across catalysts and real oils — unproven.
2. Few real-oil datasets with carbon-number-resolved product + full C/H/O + gas/aqueous balance across conditions.
3. Deactivation / lifetime prediction for continuous real-oil runs (Weber 2014, Cordero-Lanzac 2020 are partial).
4. UQ from kinetics/feed to MSP/LCA practically absent; TEAs use point yields.

Research hubs: PNNL/NREL (process design, stabilization, deactivation); Groningen/Twente (real-oil continuous HDO, polymerization vs hydrotreating); UPV/EHU Spain (lumped raw-oil kinetics + deactivation); China, Korea (reviews/catalysts, no distinct modeling school identified).

## Where our simulation fits — candidate directions

1. **Severity-parameterized yield model (most feasible).** Turn the BO-fitted per-stage conversions into functions of (T, pH2, LHSV) with a few interpretable "catalyst descriptor" multipliers (hydrogenation, C–O cleavage, acid cracking, deactivation). Catalyst-agnostic in form; refit per catalyst. Needs multi-condition data — our current fit is single-point.
2. **Lumped kinetics in RPlug/RCSTR** with C-number / boiling-cut product lumps, following Cordero-Lanzac 2020 structure + Bagnato 2019 cut mapping. Joint fit of C-number distribution, H2 uptake, C/H/O balance.
3. **Bayesian UQ to MSP/GWP.** BO already gives a posterior-ish surface; propagate parameter uncertainty → SAF yield, H2 use → MSP/LCA. This is the clearest novelty gap for a TEA/LCA group.
4. Gibbs (RGibbs) per stage as thermodynamic bound / sanity check — cheap.

## Corrections to Codex output (Crossref-checked)

- `10.1016/j.biombioe.2019.105275` is Brosowski et al. 2019 (biogenic residues), **not** "Jensen et al., Hydrotreating of bio-oil: a review". Citation fabricated — its H2-vs-DOD (Ru/C) claim unsourced; use Venderbosch 2010 instead.
- `10.1021/acs.energyfuels.1c01700`: first author Cordero-Lanzac (2021), not Wang.
- `10.1007/s12155-024-10803-x`: first author Okolie, not Adeleke.
- `10.1002/bbb.2061` (Liu & Wang): issued 2019, not 2020.
- Unverified: Snowden-Swan 2020 DOI; "Stabilization of acid-rich bio-oil…" 2021 (PubMed 33445152 only).
- Added by Claude (Crossref-verified): Venderbosch et al. 2010, J Chem Technol Biotechnol, 10.1002/jctb.2354.

## Coverage limits

- Medium reasoning, ~1.5k words per run: survey is breadth-first, not exhaustive. Missing likely: Sheu 1988 lumped kinetics, Elliott 2007 review, Sharma & Bakhshi, Genesys/RMG bio-oil networks, recent (2024–26) ML on real-oil HDO. One follow-up Codex run could target those.
- No full-text reading; claims summarize abstracts/landing pages.

## Carbon-number distribution models (added 2026-11-05)

Basis for re-framing the Weibull C#-distribution model (slides `바이오항공유_김경수_241217.pptx`): the distribution is an ensemble outcome of per-molecule growth/scission events, not a random C#.

- **Direction 1 — reinterpret Weibull.** Hazard h(n) = probability chain terminates/cleaves at n. ASF/Flory = constant termination probability (geometric); Weibull = length-dependent termination. Brown & Wohletz 1995 derive Weibull from sequential fragmentation. θ = smallest surviving skeleton, λ = characteristic product size (severity), k = size dependence of cracking.
- **Direction 2 — tie parameters to conditions.** Continuous lumping (Laxminarasimhan 1996) with k(n) rising with size and a split-yield kernel; Elizalde & Ancheyta 2010 makes those parameters functions of T and P. Same move for λ, k, θ.
- No post-2000 paper found applying continuous lumping or a Weibull/ASF C# distribution to pyrolysis-oil HDO. Closest bio-side work: Kim 2025 (lignin population balance).

### Classic (pre-2000)

| Ref | Title | DOI | OA |
|---|---|---|---|
| Flory 1936 | Molecular size distribution in linear condensation polymers | [10.1021/ja01301a016](https://doi.org/10.1021/ja01301a016) | closed |
| Anderson, Friedel & Storch 1951 | Fischer–Tropsch reaction mechanism involving stepwise growth of carbon chain | [10.1063/1.1748201](https://doi.org/10.1063/1.1748201) | closed |
| Montroll & Simha 1940 | Theory of depolymerization of long chain molecules | [10.1063/1.1750807](https://doi.org/10.1063/1.1750807) | closed |
| Brown & Wohletz 1995 | Derivation of the Weibull distribution based on physical principles and its connection to the Rosin–Rammler and lognormal distributions | [10.1063/1.360073](https://doi.org/10.1063/1.360073) | closed |
| Chou & Ho 1988 | Continuum theory for lumping nonlinear reactions | [10.1002/aic.690340914](https://doi.org/10.1002/aic.690340914) | closed |
| Laxminarasimhan, Verma & Ramachandran 1996 | Continuous lumping model for simulation of hydrocracking | [10.1002/aic.690420925](https://doi.org/10.1002/aic.690420925) | closed |

### Post-2000

| Ref | Title | Relevance | DOI | OA |
|---|---|---|---|---|
| Puskas & Hurlbut 2003 | Comments about the causes of deviations from the ASF distribution of the FT reaction products | why real C# distributions deviate from ASF; length-dependent growth probability | [10.1016/s0920-5861(03)00305-5](https://doi.org/10.1016/s0920-5861(03)00305-5) | closed |
| Tavakoli 2008 | Application of ASF equation in the product distribution of slurry phase FT synthesis with nanosized iron catalysts | fitting ASF/two-α to C# data | [10.1016/j.cej.2007.04.017](https://doi.org/10.1016/j.cej.2007.04.017) | closed |
| Ancheyta 2005 | Kinetic modeling of hydrocracking of heavy oil fractions: a review | review of lumped vs continuous vs mechanistic | [10.1016/j.cattod.2005.08.015](https://doi.org/10.1016/j.cattod.2005.08.015) | closed |
| Elizalde & Ancheyta 2009 | Application of continuous kinetic lumping modeling to moderate hydrocracking of heavy oil | continuous lumping on real feed | [10.1016/j.apcata.2009.06.018](https://doi.org/10.1016/j.apcata.2009.06.018) | closed |
| Elizalde & Ancheyta 2010 | Modeling the effect of pressure and temperature on the hydrocracking of heavy crude oil by the continuous kinetic lumping approach | **direction 2 template**: distribution params as f(T, P) | [10.1016/j.apcata.2010.04.050](https://doi.org/10.1016/j.apcata.2010.04.050) | closed |
| Elizalde & Ancheyta 2011 | On the detailed solution and application of the continuous kinetic lumping modeling to hydrocracking of heavy oils | numerical solution details | [10.1016/j.fuel.2011.03.044](https://doi.org/10.1016/j.fuel.2011.03.044) | closed |
| Elizalde & Ancheyta 2014 | Modeling catalyst deactivation during hydrocracking of atmospheric residue by using the continuous kinetic lumping model | deactivation in distribution model | [10.1016/j.fuproc.2014.02.006](https://doi.org/10.1016/j.fuproc.2014.02.006) | closed |
| Becker 2015 | Comparing hydrocracking models: continuous lumping vs. single events | when the simple distribution model is enough | [10.1016/j.fuel.2015.09.091](https://doi.org/10.1016/j.fuel.2015.09.091) | closed |
| Boosari 2017 | Application of Bayesian approach in the parameter estimation of continuous lumping kinetic model of hydrocracking process | Bayesian fit of distribution params (cf. our BO); low-tier journal | [10.4236/aces.2017.73019](https://doi.org/10.4236/aces.2017.73019) | diamond |
| Rocha & Lage 2019 | The solution of continuous kinetic lumping models using the adaptive characterization method | solver method | [10.1002/aic.16758](https://doi.org/10.1002/aic.16758) | closed |
| Laxmi Narasimhan 2004 | Relumped single-event microkinetic model for alkane hydrocracking on shape-selective catalysts | mechanistic (single-event) C# distribution | [10.1016/j.ces.2004.07.093](https://doi.org/10.1016/j.ces.2004.07.093) | closed |
| Kumar & Froment 2006 | A generalized mechanistic kinetic model for the hydroisomerization and hydrocracking of long-chain paraffins | mechanistic C# distribution, long chains | [10.1021/ie060957w](https://doi.org/10.1021/ie060957w) | closed |
| Mitsios 2009 | Single-event microkinetic model for long-chain paraffin hydrocracking and hydroisomerization on an amorphous Pt/SiO2·Al2O3 catalyst | same, amorphous bifunctional catalyst | [10.1021/ie800974q](https://doi.org/10.1021/ie800974q) | closed |
| Kim 2025 | A population balance model for lignin: tracking molecular weights, S/G ratio, and bond compositions | population balance on lignin (bio feed) | [10.1016/j.cej.2025.170474](https://doi.org/10.1016/j.cej.2025.170474) | hybrid |

## References

All DOIs Crossref-verified 2026-11-04. OA status from OpenAlex 2026-11-05: gold/hybrid/green = free copy exists; closed = needs KIST access. Plain list for Zotero "Add Item by Identifier": `dois.txt`.

### Cited in this report

| Ref | Title | DOI | OA |
|---|---|---|---|
| Zhang 2009 | Lumping kinetic model for hydrotreating of bio-oil from the fast pyrolysis of biomass | [10.1080/15567030701750523](https://doi.org/10.1080/15567030701750523) | closed |
| Cordero-Lanzac 2020 | Adaptable kinetic model for the transient and pseudo-steady states in the HDO of raw bio-oil | [10.1016/j.cej.2020.124679](https://doi.org/10.1016/j.cej.2020.124679) | green |
| Cordero-Lanzac 2021 | Advances and challenges in the valorization of bio-oil: HDO using carbon-supported catalysts | [10.1021/acs.energyfuels.1c01700](https://doi.org/10.1021/acs.energyfuels.1c01700) | closed |
| Xiao 2020 | Refinement of the kinetic model for guaiacol HDO over platinum catalysts | [10.1002/aic.16913](https://doi.org/10.1002/aic.16913) | closed |
| Morteo-Flores 2023 | First-principles microkinetic study of the catalytic HDO of guaiacol on transition metal surfaces | [10.1002/cctc.202300671](https://doi.org/10.1002/cctc.202300671) | hybrid |
| Alvarez-Majmutov 2025 | Structural-level modeling of biocrude hydrodeoxygenation | [10.1021/acs.energyfuels.5c02844](https://doi.org/10.1021/acs.energyfuels.5c02844) | hybrid |
| Chen 2024 | Machine learning-based optimization of catalytic HDO of biomass pyrolysis oil | [10.1016/j.jclepro.2024.140738](https://doi.org/10.1016/j.jclepro.2024.140738) | closed |
| Venderbosch 2010 | Stabilization of biomass-derived pyrolysis oils | [10.1002/jctb.2354](https://doi.org/10.1002/jctb.2354) | closed |
| Ma 2019 | Process simulation and economic evaluation of bio-oil two-stage hydrogenation production | [10.3390/app9040693](https://doi.org/10.3390/app9040693) | gold |
| Bagnato & Sanna 2019 | Process and techno-economic analysis for fuel and chemical production by HDO of bio-oil | [10.3390/catal9121021](https://doi.org/10.3390/catal9121021) | gold |
| Liu & Wang 2019 | Process design and evaluations for producing pyrolytic jet fuel | [10.1002/bbb.2061](https://doi.org/10.1002/bbb.2061) | closed |
| Okolie 2024 | Data-driven framework for the techno-economic assessment of SAF from pyrolysis | [10.1007/s12155-024-10803-x](https://doi.org/10.1007/s12155-024-10803-x) | hybrid |
| Farooq 2025 | Uncovering the economic potential of SAF production pathways: a meta-analysis of TEA studies | [10.1016/j.enconman.2025.120076](https://doi.org/10.1016/j.enconman.2025.120076) | hybrid |
| de Miguel Mercader 2011 (AIChE J) | Competition between hydrotreating and polymerization reactions during pyrolysis oil HDO | [10.1002/aic.12503](https://doi.org/10.1002/aic.12503) | green |
| Weber 2014 | Modeling the kinetics of deactivation of catalysts during the upgrading of bio-oil | [10.1021/ef502483t](https://doi.org/10.1021/ef502483t) | closed |

Reports without DOI:
- Jones et al. 2013, PNNL-23053 / NREL/TP-5100-61178 — fast pyrolysis and hydrotreating bio-oil pathway. https://www.nrel.gov/docs/fy14osti/61178.pdf
- Dutta et al. 2015, NREL/TP-5100-62455 — in situ / ex situ catalytic fast pyrolysis. https://www.nrel.gov/docs/fy15osti/62455.pdf
- Snowden-Swan et al. 2020 (PNNL) — feedstock composition effect on fast pyrolysis + upgrading, TEA/LCA. DOI not found; landing page: https://www.pnnl.gov/publications/effect-feedstock-composition-fast-pyrolysis-and-upgrading-transportation-fuels-techno

### Further reading (cited in Codex raw outputs only)

| Ref | Title | DOI | OA |
|---|---|---|---|
| de Miguel Mercader 2011 (EES) | HDO of pyrolysis oil fractions: process understanding and quality assessment through co-processing | [10.1039/C0EE00523A](https://doi.org/10.1039/C0EE00523A) | green |
| Luo 2024 | Catalytic hydrodeoxygenation of pyrolysis bio-oil to jet fuel: a review | [10.1007/s11708-024-0943-7](https://doi.org/10.1007/s11708-024-0943-7) | closed |
| Lee 2016 | Recent advances in the catalytic hydrodeoxygenation of bio-oil | [10.1007/s11814-016-0214-3](https://doi.org/10.1007/s11814-016-0214-3) | closed |
| Barroso-Martín 2022 | An overview of catalysts for the HDO reaction of model compounds from lignocellulosic biomass | [10.1049/rpg2.12477](https://doi.org/10.1049/rpg2.12477) | gold |
| Mäki-Arvela & Murzin 2017 | HDO of lignin-derived phenols: from fundamental studies towards industrial applications | [10.3390/catal7090265](https://doi.org/10.3390/catal7090265) | gold |
| Yang 2023 | The catalytic HDO of bio-oil for upgradation from lignocellulosic biomass | [10.1016/j.ijbiomac.2023.124773](https://doi.org/10.1016/j.ijbiomac.2023.124773) | closed |
| Subramanyam 2015 | CFD simulations of catalytic HDO of bio-oil using Pt/Al2O3 in a fixed bed reactor | [10.1039/C5RA14985A](https://doi.org/10.1039/C5RA14985A) | closed |

Excluded: `10.1016/j.biombioe.2019.105275` (wrong paper; see Corrections).
