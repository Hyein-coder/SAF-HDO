# Catalyst properties → Weibull3 parameters (decision trees)

Date: 2026-11-17 · Script: `cattree/tree.py` · Inputs: `cattree/catalyst_features.csv`, `distfit/results/weibull_laxmi.csv` · Outputs: `cattree/results/`

## 1. Question and answer

The question: which catalyst traits separate the Weibull3 parameters (shape k, location θ, scale λ) of the SimDist carbon-number distributions?

**Only λ (scale) has a signal.**
- λ sets the width and the heavy tail. Its trees beat the permutation baseline at the edge of significance:
  - permutation p = 0.052 with both feature sets
  - LOOCV R² = 0.30 (full features) and 0.13 (complete features)
- **Dropping C1 strengthens the measured-property result.** C1 copies A2's features, so it was rerun without it (n = 9):
  - full features: λ LOO R² 0.61, p = 0.010
  - complete features: λ LOO R² 0.03, p = 0.09
  - The partial-Ni-reduction reading (measured features) is therefore more robust than the composition/preparation reading.
- k and θ show no relationship. Their permutation p ≥ 0.09 in every variant. For scale, a model that always predicts the mean scores a LOO R² of about −0.23 at n = 10.
- 16 p-values were computed (4 targets × 4 variants), so a single p ≈ 0.04–0.05 could be chance. What makes the λ result credible is that every λ tree finds the same narrow group.

**Both feature sets find the same λ grouping:**

| group | entries | λ (mean) | jet C8–16 | read-out |
|---|---|---|---|---|
| narrow | C1, A1, A2, A5 | ≈ 5.8 | 81–85 % | non-PD **and** Ce-poor (x_Ce ≤ 0.146); equivalently, Ni only partly reduced (H2-TPD reduction ≤ 96.5 %) |
| broad | A3, A4, B1 (full) / A3, A4 + B1, B4 (complete) | ≈ 10 | 52–61 % (47–61 % with B4) | Ce-rich mixed oxide, Ni-free oxide, or Ni/LaOx |
| broadest | B2, B3, (B4) | ≈ 13 | 44–48 % | PD-supported Ni with fully reduced Ni |

**Two descriptions fit the narrow group equally well, and n = 10 can't tell them apart:**
- *Composition / preparation:* not precipitation–deposition and low Ce.
- *Measured:* incompletely reduced Ni, meaning Ni is held in the La–Ti(–Ce) oxide lattice, a strong metal–support interaction. All four narrow-group catalysts have a reduction degree of 37.5–93 %. Every PD-supported Ni catalyst is 100 % reduced.

The A2 vs B4 pair is the cleanest contrast in the data. They have the same composition, and preparation alone moves λ from 5.5 to 12.2. That supports the preparation/reducibility reading over composition. B4 has no TPD data, though, so this rests on one pair.

## 2. Case mapping (user-confirmed 2026-11-17)

`HDO_exp.xlsx` columns and Fig. S8 panels run one position behind the caption's labels, and Table 1's C1/C2 SimDist values are swapped.

Evidence:
- All 22 values for C10–17 and C18+ match Table 1's 150–300 °C and >300 °C fractions within 1 wt%, once the shift and the swap are applied.
- Manuscript ¶119 says only the three-step run lacks C20+. That is true only for column a.

| col | a | b | c | d | e | f | g | h | i | j | k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entry | C1 | A1 | A2 | A3 | A4 | A5 | B1 | B2 | B3 | B4 | C2 |

C1 is Pd/C → A2 material → Ru/WZr. C2 is Pd/C → Ru/WZr. **C2 (column k) is dropped**: its Weibull3 fit is degenerate, with θ at the −40 bound. That leaves n = 10.

Fixes needed in the manuscript:
- Table 1: swap the C1/C2 SimDist values.
- ¶119: swap the C1/C2 labels.
- Fig. S8: fix the caption or reorder the panels.

## 3. Features (12; composition + preparation + measured, no computed descriptors)

| block | features | source |
|---|---|---|
| composition | x_La, x_Ti, x_Ce, x_Ni (cation mole fractions) | Mixed oxides: from the formula. Supported catalysts (A5, B1–B3): 13.5 wt% Ni (S18) on the support formula mass. |
| preparation / configuration | prep_PD, Ni_supported, has_RuWZr | Table 1 |
| measured | H2_uptake, red_pct, Ni_disp | SI Table S18 (A1–A3, A5, B1–B3) |
| | BET | SI Table S1 (A1–A5) |
| | O_def (O_defect/O_L) | SI Table S16 (A1–A5, B1–B3) |

All values are in `results/features_used.csv`. Assumptions:
- C1 inherits A2's measured values, since its bed 2 is the A2 material.
- B4 has no measured data.
- A4 (Ni-free) has no S18 data.

## 4. Model and results

- `DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=0)`.
- Two kinds of tree:
  - one multi-output tree on z-scored (k, θ, λ)
  - one single-output tree per parameter, in raw units
- Validation:
  - LOOCV via `cross_val_predict`
  - `permutation_test_score`: 1000 shuffles, neg-MSE, scored by LOO
- Two feature sets, each run with and without C1 (`_noC1`, n = 9):
  - **full**: all 12 features, NaN allowed
  - **complete**: the 7 features with no gaps, as a check against missing-data artifacts

| model | full: train R² | full: LOO R² | full: perm p | complete: train R² | complete: LOO R² | complete: perm p |
|---|---|---|---|---|---|---|
| all (z) | 0.63 | −0.38 | 0.12 | 0.51 | −0.49 | 0.27 |
| k | 0.78 | −0.06 | 0.09 | 0.42 | −1.26 | 0.82 |
| θ | 0.57 | −0.34 | 0.16 | 0.26 | −1.54 | 0.96 |
| **λ** | 0.92 | **0.30** | **0.052** | 0.91 | **0.13** | **0.052** |

Without C1 (n = 9):

| model | full_noC1: LOO R² | full_noC1: perm p | complete_noC1: LOO R² | complete_noC1: perm p |
|---|---|---|---|---|
| all (z) | −0.96 | 0.51 | −0.09 | 0.036 |
| k | −1.17 | 0.57 | −0.78 | 0.43 |
| θ | −1.42 | 0.63 | −0.07 | 0.10 |
| **λ** | **0.61** | **0.010** | 0.03 | 0.09 |

λ, full_noC1 tree:

```
red_pct <= 96.5 → 5.58 {A1, A2, A5}
red_pct > 96.5 ─┬ O_def <= 0.7 → 13.24 {B2, B3, B4*}
                └ O_def > 0.7 → 9.99 {A3, A4*, B1}
```

The complete_noC1 multi-output tree splits on prep_PD, then x_Ce, and isolates the same {A1, A2, A5} group.

Split rules and leaf members are in `results/rules_<variant>.txt`, and figures in `results/trees_<variant>.png`. The variants are full, complete, full_noC1 and complete_noC1.

**λ, full feature set**

```
red_pct <= 96.5 ─┬ x_Ce <= 0.12 → 5.61  {A1, A5}
                 └ x_Ce >  0.12 → 6.04  {C1, A2}
red_pct >  96.5 ─┬ O_def <= 0.7 → 13.24 {B2, B3, B4*}
   (or missing)  └ O_def >  0.7 → 9.99  {A3, A4*, B1}
```

`*` marks rows routed by a missing value, not a measured one.

**λ, complete feature set**

```
prep_PD = 0 ─┬ x_Ce <= 0.146 → 5.82  {C1, A1, A2, A5}
             └ x_Ce >  0.146 → 10.14 {A3, A4}
prep_PD = 1 ─┬ x_La <= 0.25  → 13.75 {B2, B3}
             └ x_La >  0.25  → 10.97 {B1, B4}
```

**The BET splits that dominate the k, θ and multi-output trees in the full set are not trustworthy:**
- BET is missing for B1–B4. scikit-learn sends missing rows to whichever side lowers the error, so "BET > 4" really means "BET > 4 or unmeasured".
- The low-BET leaf {C1, A2, A5} rests on two measurements: A2 = 1 m²/g, which C1 inherits, and A5 = 3 m²/g.
- These splits disappear in the complete set, and their permutation p-values are not significant.

## 5. Caveats
- **Too few cases.** n = 10 against 7–12 features means a depth-2 tree will find some clean-looking split by chance. Only λ survives the permutation test, and only at p ≈ 0.05. Read the trees as a description of which cases group together, not as a predictive model.
- **Collinear features.** Equivalent descriptions of the same partition swap in and out (prep_PD + x_Ce vs red_pct). Read impurity importances as "one of these", not as a ranking.
- **Missing data.** Measured properties cover 6–9 of the 10 cases: BET 6, S18 8, O_def 9. Missing rows are routed silently.
- **C1 inherits A2's measured properties.** Its extra Ru/WZr bed is captured only by has_RuWZr, which no tree ever used: C1 and A2 have almost the same λ (6.54 vs 5.54). In leave-one-out, C1 and A2 are near-twins, so each is easy to predict when the other is held in training. The `_noC1` runs check this. They show it doesn't inflate the full-feature λ result (R² rises 0.30 → 0.61) but does prop up the complete-feature one (0.13 → 0.03).
- **Composition fractions are approximate.** For A5's La2Ti0.5Ce0.5Ox support, the conversion assumes O5 (Ce as CeO2) rather than about O4.75. That shifts A5's fractions by about 0.002 and changes no split.
- **Weak supporting data.** The targets are fitted parameters carrying their own uncertainty (`distfit/results/params.csv` has SEs). The SimDist input is integer wt%.

## 6. Reproduce

```
PYTHONIOENCODING=utf-8 /d/saf_hdo/.venv/Scripts/python.exe cattree/tree.py --selftest
PYTHONIOENCODING=utf-8 /d/saf_hdo/.venv/Scripts/python.exe cattree/tree.py
```
