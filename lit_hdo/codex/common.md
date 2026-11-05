CONTEXT (our project — do not run shell commands, do not read local files; use web search only):
We do TEA/LCA of sustainable aviation fuel (SAF) from biomass fast pyrolysis oil upgraded by 3-stage hydrodeoxygenation (HDO), simulated in Aspen Plus. Each HDO stage is currently an RStoic/RYield block; reaction stoichiometry/conversions were fitted by Bayesian optimization (GPyOpt) to match an experimental carbon-number distribution (C5-C20+) of the upgraded oil. Limitation: RYield/RStoic yields are fixed, so feed composition, H2 partial pressure, temperature, or LHSV changes do not propagate to product chemistry. We want to link the simulation to an HDO reaction model, ideally CATALYST-AGNOSTIC (works across NiMo/CoMo sulfided, Pt/Pd/Ru noble, Ni, etc., or parameterized by few descriptors), to make SA/optimization physically meaningful.

RULES:
- Cite only papers you actually found via web search. Give authors, year, journal, DOI for each. If a DOI cannot be confirmed, mark it [DOI unverified]. Never invent.
- Prefer 2010-2026, include seminal older work if foundational. Include reviews.
- Be concise: bullet points, tables OK. Max ~1500 words. End with a reference list.
