from aspen_utils import AspenSim
import os
import matplotlib.pyplot as plt
import pandas as pd

sim = AspenSim(r"D:\saf_hdo\aspen\Simulation_260126\260115_pyrolysis_oil_CC_case_a.bkp")
summary_coef = pd.read_excel(r"D:\saf_hdo\aspen\summary_conversion_coef_new_20260130.xlsx")

#%%
# 0. Get the carbon length of fractional conversion bases

component_list = [
    "BIOMASS", "WATER", "H2", "O2", "N2", "AR", "CO", "CO2", "CH4", "C2H4",
    "C2H4O", "C2H4O2", "C2H6", "C2H6O", "C3H6", "C3H8", "C3H8O", "C3H8O2",
    "C4H10", "C4H4O", "C4H8O", "C4H8O2", "C5H4O2", "C5H12", "C6H6", "C6H6O",
    "C6H8O", "C6H12", "C6H12O6", "C6H14", "C6H14O6", "C7H8", "C7H8O", "C7H14",
    "C7H14-CY", "C7H14-ME", "C8H10", "C8H10O", "C8H16", "C8H18", "C8H16-CY",
    "C9H10O3", "C9H18", "C9H20", "C9H20-A", "C10H22-C", "C11H14O", "C11H20-C",
    "C11H24", "C12H16O2", "C12H20-C", "C12H22O1", "C12H26", "C13H18O2",
    "C13H20O2", "C13H26", "C13H28", "C14H12", "C14H20O2", "C14H28A", "C14H30",
    "C15H32O6", "C15H30", "C16H32", "C16H34", "C17H34", "C17H36", "C17H36O6",
    "C18H38", "C19H18O5", "C19H24O2", "C19H38", "C19H40", "C20H40", "C20H42",
    "C20H22O9", "C20H42O6", "C21H24O1", "C21H42", "C22H46O2", "C22H44",
    "C23H22O6", "C23H46", "C24H42O", "C24H48", "C25H22O1", "C25H50", "C26H42O4",
    "C26H52", "C27H54", "C28H56", "C29H58", "C30H62", "SO2", "NO2", "ALDEHYDE",
    "KETONES", "ALCOHOLS", "GUAIACOL", "LMWS", "HMWS", "N2COMPOU", "SCOMPOUN",
    "S", "C", "ASH", "SIO2", "LMWLA"
]
component_to_carbon_number = {
    # "ACIDS": 4,
    "ALDEHYDE": 8,
    "KETONES": 3,
    "ALCOHOLS": 6,
    "GUAIACOL": 7,
    "LMWS": 6,
    "HMWS": 12,
    # "EXTRACTI": 20,
    "N2COMPOU": 8,
    "SCOMPOUN": 12,
    "LMWLA": 16,
    # "LMWLB": 12,
    # "HLB": 17,
}
carbon_number_list = [n for n in range(1, 26)]
carbon_number_to_component = {n: [c for c in component_list if f"C{n}" in c]
                              for n in carbon_number_list}
for c, n in component_to_carbon_number.items():
    if n in carbon_number_list:
        carbon_number_to_component[n].append(c)
carbon_number_to_component = carbon_number_to_component

all_component_to_carbon_number = {}
for n, cs in carbon_number_to_component.items():
    for c in cs:
        all_component_to_carbon_number[c] = n

basis_carbon_length = [all_component_to_carbon_number[b] for b in summary_coef['Fractional Conversion of Component']]

#%%
# 1. Sort the reactions according to the basis carbon length
summary_coef['Carbon Length of Basis'] = basis_carbon_length
summary_coef_sorted = summary_coef.sort_values(by=['Carbon Length of Basis'])

#%%
# 2. Plot the conversion coefficient of each reaction
coef_draw = summary_coef_sorted[['Carbon Length of Basis'] + [f'({case}) Fractional conversion' for case in ['A', 'F', 'I']]]
num_rxn = len(coef_draw)

fig, ax = plt.subplots(1, 1)
for case in ['A', 'F', 'I']:
    ax.plot([n for n in range(num_rxn)], coef_draw[f'({case}) Fractional conversion'], label=case)
ax.legend()
plt.show()
