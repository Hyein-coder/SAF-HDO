import os
from aspen_utils import AspenSim
from utils import CustomEvaluator
import GPy, GPyOpt
from GPyOpt.methods import ModularBayesianOptimization
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd

aspen_path = r"D:\saf_hdo\aspen\251009_pyrolysis_oil_CC.bkp"
rxtor_nodes = [
    "\Data\Blocks\R-201\Input\CONV",
    "\Data\Blocks\R-202\Input\CONV",
    "\Data\Blocks\R-203\Input\CONV",
]
sim = AspenSim(aspen_path, case_target="a", rxtor_nodes=rxtor_nodes)
save_dir = os.path.join(os.getcwd(), "results")

cc = sim.get_carbon_number_composition(sim.prod_stream)
print(cc)

plt.plot(cc.keys(), cc.values(), '-', label='Simulation')
plt.plot(sim.target.keys(), sim.target.values(), 'o', label='Experimental Data')
plt.legend()
plt.show()

import copy
r_now = sim.get_rxn_coefficients()
fig, ax = plt.subplots(1, 3, figsize=(10, 10))
for idx, r0 in enumerate(r_now):
    ax_now = ax[idx]
    ax_now.plot(range(len(r0)), r0)
plt.show()

#%%

row_names = ["Iter"]
for rxtor_idx, rxtor_indices in enumerate(sim.rxn_indices):
    for rxn_idx in rxtor_indices:
        row_names.append(f"R{rxtor_idx+1}_rxn{rxn_idx}")
for n in cc.keys():
    row_names.append(f"Prod_C{n}")

row_history = []
iter = 0
file_tag = 'perturbation' + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for rxtor_idx, rxtor_coeff in enumerate(r_now):
    for rxn_idx, rxn_coeff in enumerate(rxtor_coeff):
        r_new = copy.deepcopy(r_now)
        r_new[rxtor_idx][rxn_idx] = rxn_coeff * 0.8
        sim.apply_rxn_coefficients(r_new)
        cc = sim.get_carbon_number_composition(sim.prod_stream)

        row_history.append([iter] + [item for sublist in r_new for item in sublist] + list(cc.values()))
        df = pd.DataFrame(row_history, columns=row_names)
        df.to_excel(os.path.join(save_dir, f"{file_tag}_log.xlsx"))

        iter += 1
        # if iter > 2:
        #     break

#%%
rxn_extent = [0.8, 0.5]
row_history = []
iter = 0
file_tag = 'perturbation' + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for rxtor_idx, rxtor_coeff in enumerate(r_now):
    for e in rxn_extent:
        r_new = copy.deepcopy(r_now)
        r_new[rxtor_idx] = [r * e for r in rxtor_coeff]
        sim.apply_rxn_coefficients(r_new)
        cc = sim.get_carbon_number_composition(sim.prod_stream)

        row_history.append([iter] + [item for sublist in r_new for item in sublist] + list(cc.values()))
        df = pd.DataFrame(row_history, columns=row_names)
        df.to_excel(os.path.join(save_dir, f"{file_tag}_log.xlsx"))

        iter += 1

#%%
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
n_cc = len(cc.keys())
iter = 0
for idx in range(len(sim.rxn_indices)):
    for e in rxn_extent:
        ax_now = ax[idx]
        ax_now.plot(row_names[-n_cc:], row_history[iter][-n_cc:], '-', label=f"e={e}")
        iter += 1
plt.legend()
plt.show()