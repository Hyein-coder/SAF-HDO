import os
from aspen_utils import AspenSim
from utils import CustomEvaluator
import GPy, GPyOpt
from GPyOpt.methods import ModularBayesianOptimization
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd

aspen_path = r"D:\saf_hdo\aspen\251009_pyrolysis_oil_CC_C6H8O_DGFORM_SMR_solver.bkp"
sim = AspenSim(aspen_path, case_target="a")
save_dir = os.path.join(os.getcwd(), "results")

cc_pyro = sim.get_carbon_number_composition("207")
cc_init = sim.get_carbon_number_composition(sim.prod_stream)
print(cc_init)

plt.plot(cc_init.keys(), cc_init.values(), '-', label='Simulation')
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

row_names = ["Iter"]
for rxtor_idx, rxtor_indices in enumerate(sim.rxn_indices):
    for rxn_idx in rxtor_indices:
        row_names.append(f"R{rxtor_idx+1}_rxn{rxn_idx}")
for n in cc_init.keys():
    row_names.append(f"Prod_C{n}")

#%%
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
row_history = []
rxn_extent = [0.9, 0.7, 0.5, 0.3]
file_tag = 'perturbation' + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for iter, e in enumerate(rxn_extent):
    r_new = []
    for rxtor_coeff in r_now:
        r_new.append([rxn_coeff * e for rxn_coeff in rxtor_coeff])
    sim.apply_rxn_coefficients(r_new)
    cc = sim.get_carbon_number_composition(sim.prod_stream)

    row_history.append([iter] + [item for sublist in r_new for item in sublist] + list(cc.values()))
    df = pd.DataFrame(row_history, columns=row_names)
    df.to_excel(os.path.join(save_dir, f"{file_tag}_log.xlsx"))

#%%
rxns_C5 = [15]

file_tag = 'perturbation' + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for iter, e in enumerate(rxn_extent):
    r_new = copy.deepcopy(r_now)
    for rxn_number_aspen in rxns_C5:
        rxtor_idx = 0
        rxn_idx = sim.rxn_indices[rxtor_idx].index(rxn_number_aspen)
        r_new[rxtor_idx][rxn_idx] = r_new[rxtor_idx][rxn_idx] * e
    sim.apply_rxn_coefficients(r_new)
    cc = sim.get_carbon_number_composition(sim.prod_stream)

    row_history.append([iter] + [item for sublist in r_new for item in sublist] + list(cc.values()))
    df = pd.DataFrame(row_history, columns=row_names)
    df.to_excel(os.path.join(save_dir, f"{file_tag}_log.xlsx"))

#%% C5 형성/분해 반응들만 고려하기 -> 알고보니 C5가 아니라 C6가 문제였음
rxns_interest = [15, 16]
rxn_extent = [0.9, 0.5, 0.1]
rxn_set = [(e1, e2) for e1 in rxn_extent for e2 in rxn_extent]

file_tag = 'perturbation' + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
row_history = []
for iter, e_set in enumerate(rxn_set):
    r_new = copy.deepcopy(r_now)
    for rxn_number_aspen, e in zip(rxns_interest, e_set):
        rxtor_idx = 0
        rxn_idx = sim.rxn_indices[rxtor_idx].index(rxn_number_aspen)
        r_new[rxtor_idx][rxn_idx] = r_new[rxtor_idx][rxn_idx] * e
    sim.apply_rxn_coefficients(r_new)
    cc = sim.get_carbon_number_composition(sim.prod_stream)

    row_history.append([iter] + [item for sublist in r_new for item in sublist] + list(cc.values()))
    df = pd.DataFrame(row_history, columns=row_names)
    df.to_excel(os.path.join(save_dir, f"{file_tag}_log.xlsx"))

#%% C# >= 15
rxns_interest = [7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
rxn_extent = [0.9, 0.5, 0.3, 0.1]

file_tag = 'perturbation' + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
row_history = []
for iter, e in enumerate(rxn_extent):
    r_new = copy.deepcopy(r_now)
    for rxn_number_aspen in rxns_interest:
        rxtor_idx = 0
        rxn_idx = sim.rxn_indices[rxtor_idx].index(rxn_number_aspen)
        r_new[rxtor_idx][rxn_idx] = r_new[rxtor_idx][rxn_idx] * e
    sim.apply_rxn_coefficients(r_new)
    cc = sim.get_carbon_number_composition(sim.prod_stream)

    row_history.append([iter] + [item for sublist in r_new for item in sublist] + list(cc.values()))
    df = pd.DataFrame(row_history, columns=row_names)
    df.to_excel(os.path.join(save_dir, f"{file_tag}_log.xlsx"))

#%% 절반으로 자르는 반응들
rxn_gases = [3, 4, 5, 6, 8, 9, 10, 14, 15, 17, 18, 19, 20, 28]
rxns_interest = [12, 21, 22, 23, 24, 25, 26, 27, 29]
rxn_extent = [0.9, 0.5, 0.3, 0.1]

file_tag = 'perturbation' + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
row_history = []
for iter, e in enumerate(rxn_extent):
    r_new = copy.deepcopy(r_now)
    for rxn_number_aspen in rxn_gases:
        rxtor_idx = 0
        rxn_idx = sim.rxn_indices[rxtor_idx].index(rxn_number_aspen)
        r_new[rxtor_idx][rxn_idx] = min(r_new[rxtor_idx][rxn_idx] * 2, .99)
    for rxn_number_aspen in rxns_interest:
        rxtor_idx = 0
        rxn_idx = sim.rxn_indices[rxtor_idx].index(rxn_number_aspen)
        r_new[rxtor_idx][rxn_idx] = r_new[rxtor_idx][rxn_idx] * e
    sim.apply_rxn_coefficients(r_new)
    cc = sim.get_carbon_number_composition(sim.prod_stream)

    row_history.append([iter] + [item for sublist in r_new for item in sublist] + list(cc.values()))
    df = pd.DataFrame(row_history, columns=row_names)
    df.to_excel(os.path.join(save_dir, f"{file_tag}_log.xlsx"))

#%% Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
n_cc = len(cc_init.keys())

ax_now = ax
ax_now.plot(cc_init.keys(), cc_init.values(), '-', label='Initial')
for row_i in row_history:
    # ax_now.plot(cc_init.keys(), row_i[-n_cc:], '-o', label=f"{rxn_set[int(row_i[0])]}")
    ax_now.plot(cc_init.keys(), row_i[-n_cc:], '-', label=f"{rxn_extent[int(row_i[0])]}")
# plt.legend()
# plt.show()

data_targets = sim.data
case1 = ['a', 'b', 'c', 'f']
case2 = ['d', 'e', 'g', 'h', 'i', 'j', 'k']
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for case in case2:
    _, data_case = sim.set_target(case)
    ax_now.plot(data_case.keys(), data_case.values(), '-o', label=case)
plt.legend()
plt.savefig(os.path.join(save_dir, f"{file_tag}_result.png"))
plt.show()

#%%
case1 = ['a', 'b', 'c', 'f']
case2 = ['d', 'e', 'g', 'h', 'i', 'j', 'k']
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plt.plot(cc_pyro.keys(), cc_pyro.values(), '-k', label='Pyrolysis')
for case in case1:
    _, data_case = sim.set_target(case)
    plt.plot(data_case.keys(), data_case.values(), '-^', label=case)
for case in case2:
    _, data_case = sim.set_target(case)
    plt.plot(data_case.keys(), data_case.values(), '--s', label=case)
plt.xlim(5, 26)
plt.legend()
plt.show()

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for case in data_targets.columns[1:]:
    _, data_case = sim.set_target(case)
    ax.plot(data_case.keys(), data_case.values(), '-', label=case)
plt.legend()
plt.show()