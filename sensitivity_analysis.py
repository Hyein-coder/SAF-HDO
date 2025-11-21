from aspen_utils import AspenSim
import os
import matplotlib.pyplot as plt

save_dir = os.path.join(os.getcwd(), 'results')
simA = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_a_rxn_index.bkp")
simI = AspenSim(r"D:\saf_hdo\aspen\251111_pyrolysis_oil_CC_case_i_indexing.bkp", case_target="i")
# simK = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_k_rxn_enabled.bkp", case_target="k")
simF = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_f.bkp", case_target="f")

sims = [simA, simI, simF]
original_coefs = []
for s in sims:
    original_coefs.append(s.get_rxn_coefficients())

#%
sim_main = AspenSim(r"D:\saf_hdo\aspen\caseSA_DS_2.bkp")
coef_main = sim_main.get_rxn_coefficients()

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

#% Show the case data coverage
import matplotlib.cm as cm

targets = {}
for a in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
    _, tt = simF.set_target(a)
    targets[a] = tt
cmap_target = cm.get_cmap('viridis', len(targets))

fig, ax = plt.subplots(1, 1)
for i, (key, val) in enumerate(targets.items()):
    plt.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
             color=cmap_target(i))
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

#%
print("=== Check If Original Coefficients Match the Data ===")
fig, axs = plt.subplots(1, 2, figsize=(8, 5))
ax_hdo = axs[0]
ax_coef = axs[1]
colors = ['#428bca', '#d9534f', '#5cb85c', '#faa632']
markers = ['o', 's', '^', 'x']
for i, (key, val) in enumerate(targets.items()):
    ax_hdo.plot(val.keys(), val.values(), 'o', markersize=3, color=cmap_target(i))
original_results = []
for n, c, cl, mk in zip(names, original_coefficients, colors, markers):
    hdo_res, _ = sim_main.apply_rxn_coefficients(c)
    original_results.append(hdo_res)

    _, hdo_target = sim_main.set_target(n)
    ax_hdo.plot(hdo_res.keys(), hdo_res.values(), label=n+" (sim)", color=cl)
    ax_hdo.plot(hdo_target.keys(), hdo_target.values(), mk, label=n+" (exp)", color=cl)
    ax_coef.plot(range(len(c[0])), c[0], label=n, color=cl, marker=mk)
ax_hdo.set_title("HDO Product Composition")
ax_hdo.set_xlabel("Carbon Lengths")
ax_hdo.legend()

ax_coef.set_title("Reaction Coefficients")
ax_coef.set_xlabel("Reaction Index")
fig.tight_layout()
plt.show()
#%
def rxn_coef_interp(_x, _y):
    rxn_coef = []
    for a_rxtor, i_rxtor, f_rxtor in zip(*original_coefficients):
        rxn_coef_rxtor = []
        for a, i, f in zip(a_rxtor, i_rxtor, f_rxtor):
            rxn_coef_rxtor.append(a * (1-_x-_y) + i * _x + f * _y)
        rxn_coef.append(rxn_coef_rxtor)
    return rxn_coef

original_points = [(0, 0), (1, 0), (0, 1)]
hdo_orig, _ = sim_main.apply_rxn_coefficients(original_coefficients[0])

#%% Random data generation
import numpy as np
import pandas as pd
import datetime
np.random.seed(42)

N = 20
noise = True
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

dir_all = os.path.join(r"D:\saf_hdo\aspen", f'sa_{timestamp}_noise_{noise}')
dir_conv = os.path.join(dir_all, 'converged')
dir_error = os.path.join(dir_all, 'error')
os.makedirs(dir_conv, exist_ok=True)
os.makedirs(dir_error, exist_ok=True)

def res2df(_interp_point, _coef, _res, _stat):
    row_interp = {}
    for i, p in enumerate(_interp_point):
        row_interp[f"Interp{i+1}"] = p
    row_coef = {}
    for i, c_rxtor in enumerate(_coef):
        for idx, c in zip(sim_main.rxn_indices[i], c_rxtor):
            row_coef[f"R{i+1}_rxn{idx}"] = c
    row_res = {}
    for i, r in _res.items():
        row_res[f"C{i}"] = r
    row = {**row_interp, **row_coef, **row_res, 'state': _stat}
    return row

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
    print(f"Random Sensitivity Analysis #{idx}/{N}")
    coef_sa = rxn_coef_interp(x, y)
    if noise:
        coef_sa += np.random.normal(0, 0.2, len(coef_sa))
        coef_sa = [[min(max(c, 0), 1) for c in cc] for cc in coef_sa]

    res_sa, stat = sim_main.apply_rxn_coefficients(coef_sa)

    sa_coefficients.append(coef_sa)
    sa_results.append(res_sa)
    sa_all_rows.append(res2df((x, y), coef_sa, res_sa, stat))
    if res_sa is not None:
        if stat == 'Error':
            sim_main.save_simulation_as(os.path.join(dir_error, f'caseSA_{idx}.bkp'))
        else:
            sim_main.save_simulation_as(os.path.join(dir_conv, f'caseSA_{idx}.bkp'))

df_sa = pd.DataFrame(sa_all_rows)
df_sa.to_csv(os.path.join(dir_all, 'results.csv'))
#%
fig, ax = plt.subplots(1, 1)
cmap_rand = cm.get_cmap('magma', N)
cmap_original = cm.get_cmap('viridis', len(original_points))

df_sa_converged = df_sa[df_sa['state'].isin(['Converged', 'Warning'])]
product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

for i, (key, val) in enumerate(targets.items()):
    plt.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
             color=cmap_target(i))
for i, row in df_sa_converged.iterrows():
    res = row[col_products].tolist()
    if res is None:
        continue
    plt.plot(product_carbon_range, res, '-', color=cmap_rand(i))
for i, res in enumerate(original_results):
    plt.plot(res.keys(), res.values(), '-', color=cmap_original(i))

ax.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(dir_all, 'plot.png'))
plt.show()

#%
num_fail = df_sa[df_sa['state'] == 'RxnCoefError'].shape[0]
num_error = df_sa[df_sa['state'] == 'Error'].shape[0]
print(f"Simulation failed: {num_fail + num_error} / {N}")
none_indices = [i for i, x in enumerate(sa_results) if x is None]

#%%
import time
print("=== Effect of Parameters ===")
N_grid = 11
grid_values = np.linspace(0, 1, N_grid)
grid_points = [[(x, 0) for x in grid_values], [(0, x) for x in grid_values]]

param_idx = 1
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dir_all = os.path.join(r"D:\saf_hdo\aspen", f'grid_{timestamp}_param_{param_idx}')
dir_conv = os.path.join(dir_all, 'converged')
dir_error = os.path.join(dir_all, 'error')
os.makedirs(dir_conv, exist_ok=True)
os.makedirs(dir_error, exist_ok=True)

sa_points = grid_points[param_idx]

sa_coefficients = []
sa_results = []
sa_all_rows = []
for idx, (x, y) in enumerate(sa_points):
    print(f"Grid for Parameter {param_idx}: #{idx}/{N_grid}")
    coef_sa = rxn_coef_interp(x, y)
    if noise:
        coef_sa += np.random.normal(0, 0.05, len(coef_sa))
        coef_sa = [[min(max(c, 0), 1) for c in cc] for cc in coef_sa]

    res_sa, stat = sim_main.apply_rxn_coefficients(coef_sa)

    if stat == 'Error':
        time.sleep(2)
        res_sa, stat = sim_main.apply_rxn_coefficients(coef_sa)

    sa_coefficients.append(coef_sa)
    sa_results.append(res_sa)
    sa_all_rows.append(res2df((x, y), coef_sa, res_sa, stat))
    if res_sa is not None:
        if 'Error' in stat:
            sim_main.save_simulation_as(os.path.join(dir_error, f'caseSA_{idx}.bkp'))
        else:
            sim_main.save_simulation_as(os.path.join(dir_conv, f'caseSA_{idx}.bkp'))

df_sa = pd.DataFrame(sa_all_rows)
df_sa.to_csv(os.path.join(dir_all, 'results.csv'))
#%
fig, ax = plt.subplots(1, 1)
cmap_rand = cm.get_cmap('magma', N+3)
cmap_original = cm.get_cmap('viridis', len(original_points))

df_sa_converged = df_sa[df_sa['state'].isin(['Converged', 'Warning'])]
product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

for i, (key, val) in enumerate(targets.items()):
    plt.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
             color=cmap_target(i))
for i, row in df_sa_converged.iterrows():
    res = row[col_products].tolist()
    if res is None:
        continue
    plt.plot(product_carbon_range, res, '-', color=cmap_rand(i))

x_ticks = np.arange(5, product_carbon_range[-1] + 1, 5)
ax.set_xticks(x_ticks)

ax.grid()
ax.set_xlabel("Carbon Length")
ax.set_ylabel("Weight Fraction in HDO Product")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(dir_all, 'plot.png'))
plt.show()

num_fail = df_sa[df_sa['state'] == 'RxnCoefError'].shape[0]
num_error = df_sa[df_sa['state'] == 'Error'].shape[0]
print(f"Simulation failed: {num_fail + num_error} / {N}")
none_indices = [i for i, x in enumerate(sa_results) if x is None]
