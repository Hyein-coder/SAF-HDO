import pandas as pd
import os
import re

# Define a function to extract the number from the column name
def extract_tag(name):
    # Regex pattern: look for an underscore followed by digits at the end of the string
    match = re.search(r'_(\d+)$', name)
    if match:
        return int(match.group(1))
    return None # Return None if pattern doesn't match

read_folder = r"D:\saf_hdo\aspen\sa_20251203"
read_files = os.listdir(os.path.join(read_folder, "converged"))
read_index = [extract_tag(s[:-4]) for s in read_files]
read_df = pd.read_csv(os.path.join(read_folder, 'converged_results_combined.csv'))
read_df['case_num'] = read_index
read_df = read_df.set_index('case_num').sort_index()

df_sa = pd.read_csv(os.path.join(read_folder, "params_retrieved.csv"))
df_sa_converged = df_sa.iloc[read_df.index.tolist()]

read_df['Interp1'] = df_sa_converged['Interp1']
read_df['Interp2'] = df_sa_converged['Interp2']
read_df.rename({'Interp1': 'param_0', 'Interp2': 'param_1'}, axis=1, inplace=True)
read_df.to_csv(os.path.join(read_folder, "results_random_final.csv"))
#%%


#%%
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from aspen_utils import AspenSim

simA = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_a_3.bkp")
simI = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_i.bkp", case_target="i")
# simK = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_k_rxn_enabled.bkp", case_target="k")
simF = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_f.bkp", case_target="f")

sims = [simA, simI, simF]
original_coefs = []
for s in sims:
    original_coefs.append(s.get_rxn_coefficients())

sim_main = simA
def convert_coef(simX):
    coef_X = simX.get_rxn_coefficients()
    coef_X_for_main = [[0 for _ in rxn_indices] for rxn_indices in sim_main.rxn_indices]
    for rxtor, rxn_indices in enumerate(sim_main.rxn_indices):
        for idx, rxn_no in enumerate(rxn_indices):
            if rxn_no in simX.rxn_indices[rxtor]:
                coef_X_for_main[rxtor][idx] = coef_X[rxtor][simX.rxn_indices[rxtor].index(rxn_no)]
    return coef_X_for_main

names = [s.case_target for s in sims]
original_coefficients = [convert_coef(s) for s in sims]

#%
def rxn_coef_interp(_x, _y):
    rxn_coef = []
    for a_rxtor, i_rxtor, f_rxtor in zip(*original_coefficients):
        rxn_coef_rxtor = []
        for a, i, f in zip(a_rxtor, i_rxtor, f_rxtor):
            rxn_coef_rxtor.append(a * (1-_x-_y) + i * _x + f * _y)
        rxn_coef.append(rxn_coef_rxtor)
    return rxn_coef

def res2df(_interp_point, _coef, _res, _stat):
    row_interp = {}
    for i, p in enumerate(_interp_point):
        row_interp[f"Interp{i + 1}"] = p
    row_coef = {}
    for i, c_rxtor in enumerate(_coef):
        for idx, c in zip(sim_main.rxn_indices[i], c_rxtor):
            row_coef[f"R{i + 1}_rxn{idx}"] = c
    row_res = {}
    for i, r in _res.items():
        row_res[f"C{i}"] = r
    row = {**row_interp, **row_coef, **row_res, 'state': _stat}
    return row

#%%
import numpy as np

np.random.seed(42)

N = 200
noise = False

print(f"=== Random {noise} Sensitivity Analysis ===")
n_ref = len(sims)
sa_points = np.random.rand(N, n_ref-1)
for idx, p in enumerate(sa_points):
    if sum(p) > 1:
        sa_points[idx] = sa_points[idx] / sum(sa_points[idx])

sa_coefficients = []
sa_results = []
sa_all_rows = []
for idx, (x, y) in enumerate(sa_points):
    print(f"Random SA #{idx}/{N}-----")
    coef_sa = rxn_coef_interp(x, y)
    if noise:
        coef_sa += np.random.normal(0, 0.2, len(coef_sa))
    coef_sa = [[min(max(c, 0), 1) for c in cc] for cc in coef_sa]
    sa_coefficients.append(coef_sa)
    sa_all_rows.append(res2df((x, y), coef_sa, {}, None))

df_sa = pd.DataFrame(sa_all_rows)
df_sa.to_csv(os.path.join(read_folder, "params_retrieved.csv"))
