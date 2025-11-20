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
sim_main = simF
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
coefs = [convert_coef(s) for s in sims]

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

#%%
fig, axs = plt.subplots(1, 2, figsize=(8, 5))
ax_hdo = axs[0]
ax_coef = axs[1]
colors = ['#428bca', '#d9534f', '#5cb85c', '#faa632']
markers = ['o', 's', '^', 'x']
for i, (key, val) in enumerate(targets.items()):
    ax_hdo.plot(val.keys(), val.values(), 'o', markersize=3, color=cmap_target(i))
for n, c, cl, mk in zip(names, coefs, colors, markers):
    hdo_res = sim_main.apply_rxn_coefficients(c)
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
#%%
def rxn_coef_interp(_x, _y):
    rxn_coef = []
    for a_rxtor, i_rxtor, f_rxtor in zip(*coefs):
        rxn_coef_rxtor = []
        for a, i, f in zip(a_rxtor, i_rxtor, f_rxtor):
            rxn_coef_rxtor.append(a * (1-_x-_y) + i * _x + f * _y)
        rxn_coef.append(rxn_coef_rxtor)
    return rxn_coef

hdo_orig = sim_main.apply_rxn_coefficients(coefs[0])

#%% Random data generation
import numpy as np
import random
np.random.seed(42)
rand_dir = os.path.join(r"D:\saf_hdo\aspen", 'random')
os.makedirs(rand_dir, exist_ok=True)

n_ref = len(sims)
N = 100

sa_points = np.random.rand(N, n_ref-1)
for idx, p in enumerate(sa_points):
    if sum(p) > 1:
        sa_points[idx] = sa_points[idx] / sum(sa_points[idx])

sa_coefficients = []
sa_results = []
noise = True
for idx, (x, y) in enumerate(sa_points):
    print(f"Random Sensitivity Analysis #{idx}/{N}")
    coef_sa = rxn_coef_interp(x, y)
    if noise:
        coef_sa += np.random.normal(0, 0.05, len(coef_sa))
        coef_sa = [[min(max(c, 0), 1) for c in cc] for cc in coef_sa]

    res_sa = sim_main.apply_rxn_coefficients(coef_sa)


    sa_coefficients.append(coef_sa)
    sa_results.append(res_sa)
    if res_sa is not None:
        sim_main.save_simulation_as(os.path.join(rand_dir, f'caseSA_{idx}.bkp'))

#%%
original_points = [(0, 0), (1, 0), (0, 1)]
original_coefficients_retreived = []
original_results = []
for idx, (x, y) in enumerate(original_points):
    print(f"Original Data Generation #{idx}/3")
    coef = rxn_coef_interp(x, y)
    res = sim_main.apply_rxn_coefficients(coef)

    original_coefficients_retreived.append(coef)
    original_results.append(res)

#%%
fig, ax = plt.subplots(1, 1)
cmap_rand = cm.get_cmap('magma', N)
cmap_original = cm.get_cmap('viridis', len(original_points))

for i, (key, val) in enumerate(targets.items()):
    plt.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
             color=cmap_target(i))
for i, res in enumerate(sa_results):
    if res is None:
        continue
    plt.plot(res.keys(), res.values(), '-', color=cmap_rand(i))
for i, res in enumerate(original_results):
    plt.plot(res.keys(), res.values(), '-', color=cmap_original(i))

ax.grid()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

#%%
num_fail = sa_results.count(None)
print(f"Simulation failed: {num_fail} / {N}")
none_indices = [i for i, x in enumerate(sa_results) if x is None]

#%% Matching case f
# res = sim_main.get_carbon_number_composition(sim_main.prod_stream)
#
# fig, ax = plt.subplots(1, 1)
# for i, (key, val) in enumerate(targets.items()):
#     if key == 'f':
#         plt.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
#                  color=cmap_target(i))
# plt.plot(res.keys(), res.values(), '-', color='r')
# ax.grid()
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()
# plt.show()
