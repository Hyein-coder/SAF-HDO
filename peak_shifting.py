from aspen_utils import AspenSim
import os
import matplotlib.pyplot as plt
import pandas as pd

sim = AspenSim(r"D:\saf_hdo\aspen\251202_case_a_sequential25_3.bkp")
summary_coef = pd.read_excel(r"D:\saf_hdo\aspen\summary_conversion_coef.xlsx")

#%%
res = sim.get_carbon_number_composition(sim.prod_stream)
target = sim.target
plt.plot(res.keys(), res.values(), label='Pyrolysis')
plt.plot(target.keys(), target.values(), 'o', label='Experimental Data')
plt.show()
#%%
coef_base = summary_coef[['Rxn No.', '(A) Fractional conversion', '(F) Fractional conversion', '(I) Fractional conversion']]
coef_base = coef_base.set_index('Rxn No.').sort_index()

# sim.apply_rxn_coefficients([coef_base['(A) Fractional conversion'].tolist()])

rxn_c25_break = [33, 34, 35, 36, 37, 43, 44]
case_names = ['A', 'F', 'I']
hdo_res_wo_C25_break = []
coef_wo_C25_break = []

for c in case_names:
    coef = coef_base[f'({c}) Fractional conversion'].copy()
    # Without C25 breakdown
    # for rxn_no in rxn_c25_break: coef[rxn_no] = 0
    coef_wo_C25_break.append(coef.tolist())
    coef_apply = [coef.tolist()]
    hdo_res, stat = sim.apply_rxn_coefficients(coef_apply)
    if stat == 'Error':
        hdo_res, stat2 = sim.apply_rxn_coefficients(coef_apply)
    hdo_res_wo_C25_break.append(hdo_res)

#%%
pyro_c_length = sim.get_carbon_number_composition(sim.prod_stream)
# plt.plot(pyro_c_length.keys(), pyro_c_length.values(), '-o', label='Pyrolysis')
for c, res in zip(case_names, hdo_res_wo_C25_break):
    _, target = sim.set_target(f"{c.lower()}")
    plt.plot(res.keys(), res.values(), '-', label=c)
    plt.plot(target.keys(), target.values(), '^', label=c)
# plt.title("Carbon Chain Length Distribution in Pyrolysis Product")
plt.legend()
plt.xlabel("Number of Carbons")
plt.ylabel("Weight Fraction")
plt.show()

#%%
res = sim.get_carbon_number_composition(sim.prod_stream)
_, target = sim.set_target("f")
plt.plot(res.keys(), res.values(), '-')
plt.plot(target.keys(), target.values(), 'o')
plt.show()
