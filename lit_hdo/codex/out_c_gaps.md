## 1. Generalizable insights

- **Oxygen removal and carbon retention compete.** Removing oxygen as water through HDO generally retains more carbon but consumes more hydrogen. Decarboxylation and decarbonylation can reduce hydrogen demand, while losing carbon as CO₂ or CO. The trade-off depends on feed chemistry and reaction pathways; it is not a single universal yield curve. [Cordero-Lanzac et al., 2021](https://doi.org/10.1021/acs.energyfuels.1c01700); [Luo et al., 2024](https://doi.org/10.1007/s11708-024-0943-7)

- **Reactivity varies by functional group, but rankings are conditional.** Alcohols and aliphatic ethers are often readily converted; carbonyls and acids can react through several routes; phenolics and furanic structures are often more resistant. Catalyst, support acidity, steric structure, and conditions can change the dominant route, so a single ordering should not be treated as universal. [Barroso-Martín et al., 2022](https://doi.org/10.1049/rpg2.12477); [Mäki-Arvela & Murzin, 2017](https://doi.org/10.3390/catal7090265)

- **Stabilization ahead of severe treatment addresses competing chemistry.** Reactive sugars, aldehydes, and ketones can condense or polymerize, forming deposits and reducing catalyst life. Mild hydrogenation before hotter, deeper upgrading is a recurring process rationale. The underlying stabilization chemistry and best operating window remain feed- and catalyst-specific. [Weber, Olarte & Wang, 2015](https://doi.org/10.1021/ef502483t); [NREL process design report](https://www.nrel.gov/docs/fy14osti/61178.pdf)

- **Temperature and hydrogen pressure shift selectivity.** Higher temperature can improve conversion and favor cracking, decarbonylation, or decarboxylation; higher hydrogen pressure tends to favor hydrogenation and hydrogen-consuming HDO pathways. Water, phase behavior, and mass transfer can also affect observed performance. These are directional tendencies, not catalyst-independent quantitative laws. [Luo et al., 2024](https://doi.org/10.1007/s11708-024-0943-7); [Subramanyam et al., 2015](https://doi.org/10.1039/C5RA14985A)

- **Staging separates different objectives.** A lower-severity stabilization step can limit polymerization and deposits; later, hotter stages can target deeper deoxygenation and conversion of more refractory species. This supports staged design, though the appropriate number of stages and conditions need data for the actual feed and catalyst. [NREL process design report](https://www.nrel.gov/docs/fy14osti/61178.pdf); [Cordero-Lanzac et al., 2020](https://doi.org/10.1016/j.cej.2020.124679)

## 2. Key gaps

- **Transferable kinetics:** Real-oil kinetic models exist, including a seven-lump model for CoMo on activated carbon, but they remain tied to particular feed, catalyst, and operating conditions. Evidence does not yet support one kinetic parameter set spanning sulfided NiMo/CoMo, noble metals, and Ni catalysts. [Cordero-Lanzac et al., 2020](https://doi.org/10.1016/j.cej.2020.124679)

- **Real-feed and product resolution:** Studies often report bulk oxygen, liquid yield, or broad chemical families. Fewer provide matched feed/product data with reliable carbon-number distributions, elemental balances, gas and aqueous products, and conditions needed to fit and independently validate an SAF-oriented model. This limits validation of product-range predictions.

- **Deactivation and scale-up:** Deposit formation, selective catalyst deactivation, and transient-to-steady behavior are modeled in some studies, but predictive lifetime and regeneration models for continuous processing of variable real oils remain limited. Reactor-scale heat, mass transfer, and phase behavior complicate extrapolation. [Weber, Olarte & Wang, 2015](https://doi.org/10.1021/ef502483t); [Cordero-Lanzac et al., 2020](https://doi.org/10.1016/j.cej.2020.124679)

- **Uncertainty and TEA/LCA linkage:** HDO models rarely carry uncertainty in feed composition, fitted kinetics, catalyst life, and product yields through to MSP and life-cycle results. Many process assessments therefore rely on fixed conversions or yields whose uncertainty and condition dependence are not explicit.

## 3. Where research is concentrated

- **United States — PNNL and NREL:** PNNL work emphasizes stabilization, hydrotreating, catalyst deactivation, and process design; NREL contributes process-design and techno-economic system studies. [PNNL deactivation study](https://doi.org/10.1021/ef502483t); [NREL process design report](https://www.nrel.gov/docs/fy14osti/61178.pdf)

- **Netherlands — Groningen and Twente:** Groningen/Twente work emphasizes continuous hydrotreatment of real pyrolysis oils, fractionation, polymerization versus hydrotreating, and refinery co-processing. [Mercader et al., 2011](https://doi.org/10.1039/C0EE00523A); [Mercader et al., 2011](https://doi.org/10.1002/aic.12503)

- **Spain — University of the Basque Country and collaborating groups:** Recent work emphasizes lumped reaction networks for raw-oil HDO, transient operation, and catalyst deactivation. [Cordero-Lanzac et al., 2020](https://doi.org/10.1016/j.cej.2020.124679)

- **Korea:** Review and experimental contributions cover catalyst systems, mild stabilization, and HDO pathways. The sources found here establish activity, but are not enough to characterize a distinct national modeling focus. [Lee et al., 2016](https://doi.org/10.1007/s11814-016-0214-3); [“Stabilization of acid-rich bio-oil by catalytic mild hydrotreating,” 2021](https://pubmed.ncbi.nlm.nih.gov/33445152/)

- **China:** Recent reviews include bio-oil HDO for jet-fuel applications and catalyst/condition effects. A review alone does not establish the relative scale of regional research activity. [Luo et al., 2024](https://doi.org/10.1007/s11708-024-0943-7); [Yang et al., 2023](https://doi.org/10.1016/j.ijbiomac.2023.124773)

## 4. Opportunities for an Aspen + Bayesian-optimization TEA/LCA group

- **Build a severity-based yield model:** Replace fixed RStoic/RYield outputs with a compact model that predicts product-family and carbon-number yields from feed descriptors, temperature, hydrogen partial pressure, and residence time. Keep elemental and carbon balances explicit; distinguish HDO from carbon-losing routes.

- **Use catalyst descriptors as adjustable inputs:** Rather than claim universal catalyst kinetics, represent catalyst effects with a small set of interpretable parameters—such as hydrogenation, C–O cleavage, acid cracking, and deactivation tendencies—and test transfer across catalyst classes.

- **Fit and validate against richer targets:** Retain carbon-number matching, but jointly fit gas/liquid/aqueous yields, elemental balances, oxygen removal, and hydrogen consumption. Use held-out conditions or catalysts to test whether the model generalizes beyond the conditions used for fitting.

- **Propagate uncertainty to decisions:** Use Bayesian parameter distributions and alternative reaction-network structures to propagate uncertainty into SAF-range yield, hydrogen use, MSP, and LCA results. Identify which measurements would most reduce decision uncertainty.

- **Link kinetics to staged Aspen flowsheets:** Couple severity-dependent reaction outputs to stage temperature, pressure, and LHSV so process sensitivity and optimization reflect chemistry, while reporting the model’s validated operating domain.

## References

- Barroso-Martín, J. M. et al. (2022). “An overview of catalysts for the hydrodeoxygenation reaction of model compounds from lignocellulosic biomass.” *IET Renewable Power Generation*. DOI: 10.1049/rpg2.12477.
- Cordero-Lanzac, T. et al. (2020). “Adaptable kinetic model for the transient and pseudo-steady states in the hydrodeoxygenation of raw bio-oil.” *Chemical Engineering Journal*, 400, 124679. DOI: 10.1016/j.cej.2020.124679.
- Cordero-Lanzac, T. et al. (2021). “Advances and Challenges in the Valorization of Bio-Oil: Hydrodeoxygenation Using Carbon-Supported Catalysts.” *Energy & Fuels*, 35, 17008–17031. DOI: 10.1021/acs.energyfuels.1c01700.
- Lee, H. et al. (2016). “Recent advances in the catalytic hydrodeoxygenation of bio-oil.” *Korean Journal of Chemical Engineering*, 33, 3299–3315. DOI: 10.1007/s11814-016-0214-3.
- Luo, Z. et al. (2024). “Catalytic hydrodeoxygenation of pyrolysis bio-oil to jet fuel: A review.” *Frontiers of Energy*, 18, 550–582. DOI: 10.1007/s11708-024-0943-7.
- Mäki-Arvela, P. & Murzin, D. Y. (2017). “Hydrodeoxygenation of Lignin-Derived Phenols: From Fundamental Studies towards Industrial Applications.” *Catalysts*, 7, 265. DOI: 10.3390/catal7090265.
- Mercader, F. de M. et al. (2011). “Hydrodeoxygenation of pyrolysis oil fractions: process understanding and quality assessment through co-processing in refinery units.” *Energy & Environmental Science*, 4, 985–997. DOI: 10.1039/C0EE00523A.
- Mercader, F. de M. et al. (2011). “Competition Between Hydrotreating and Polymerization Reactions During Pyrolysis Oil Hydrodeoxygenation.” *AIChE Journal*, 57. DOI: 10.1002/aic.12503.
- Subramanyam, M. D., Gollakota, A. R. K. & Kishore, N. (2015). “CFD simulations of catalytic hydrodeoxygenation of bio-oil using Pt/Al₂O₃ in a fixed bed reactor.” *RSC Advances*, 5, 90354–90366. DOI: 10.1039/C5RA14985A.
- Weber, R. S., Olarte, M. V. & Wang, H. (2015). “Modeling the Kinetics of Deactivation of Catalysts during the Upgrading of Bio-Oil.” *Energy & Fuels*, 29, 273–277. DOI: 10.1021/ef502483t.
- Yang, Y. et al. (2023). “The catalytic hydrodeoxygenation of bio-oil for upgradation from lignocellulosic biomass.” *International Journal of Biological Macromolecules*, 242, 124773. DOI: 10.1016/j.ijbiomac.2023.124773.
- “Stabilization of acid-rich bio-oil by catalytic mild hydrotreating.” (2021). DOI: [DOI unverified].
- NREL. *Process Design and Economics for the Conversion of Lignocellulosic Biomass to Hydrocarbon Fuels: Fast Pyrolysis and Hydrotreating Bio-Oil Pathway* (process design report; DOI not listed).