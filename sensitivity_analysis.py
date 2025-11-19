from aspen_utils import AspenSim
import os
import matplotlib.pyplot as plt

save_dir = os.path.join(os.getcwd(), 'results')
simA = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_a_rxn_index.bkp")
simI = AspenSim(r"D:\saf_hdo\aspen\251111_pyrolysis_oil_CC_case_i_indexing.bkp", case_target="i")
simK = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_f.bkp", case_target="f")

sims = [simA, simI, simK]
original_coefs = []
for s in sims:
    original_coefs.append(s.get_rxn_coefficients())

#%
sim_main = simK
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

#%%
fig, axs = plt.subplots(1, 2, figsize=(8, 5))
ax_hdo = axs[0]
ax_coef = axs[1]
colors = ['#428bca', '#d9534f', '#5cb85c']
markers = ['o', 's', '^']
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
#%
def rxn_coef_interp(x, y):
    rxn_coef = []
    for a_rxtor, i_rxtor, k_rxtor in zip(*coefs):
        rxn_coef_rxtor = []
        for a, i, k in zip(a_rxtor, i_rxtor, k_rxtor):
            rxn_coef_rxtor.append(a * (1-x-y) + i * x + k * y)
        rxn_coef.append(rxn_coef_rxtor)
    return rxn_coef

hdo_orig = sim_main.apply_rxn_coefficients(coefs[0])

#%% Show the case data coverage
import matplotlib.cm as cm

targets = {}
for a in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
    _, tt = simK.set_target(a)
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
coarse_test = [0.1, 0.4, 0.7, 1.0]
hdo_vary_x = []
hdo_vary_y = []
sensitivity_dir = os.path.join(r"D:\saf_hdo\aspen", 'sensitivity')
os.makedirs(sensitivity_dir, exist_ok=True)
for i in coarse_test:
    coef_vary_x = rxn_coef_interp(i, 0)
    hdo_vary_x.append(sim_main.apply_rxn_coefficients(coef_vary_x))
    sim_main.save_simulation_as(os.path.join(sensitivity_dir, f'caseI_{i:.1f}.bkp'))

for i in coarse_test:
    coef_vary_y = rxn_coef_interp(0, i)
    hdo_vary_y.append(sim_main.apply_rxn_coefficients(coef_vary_y))
    sim_main.save_simulation_as(os.path.join(sensitivity_dir, f'caseK_{i:.1f}.bkp'))

#%%
fig, axs = plt.subplots(1, 2, figsize=(8,5))
cmap_coarse = cm.get_cmap('viridis', len(coarse_test)+1)

for ax, hdo in zip(axs, [hdo_vary_x, hdo_vary_y]):
    for i, (key, val) in enumerate(targets.items()):
        ax.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
                 color=cmap_target(i))
    for idx, d in enumerate([hdo_orig] + hdo):
        ax.plot(d.keys(), d.values(), '-', color=cmap_coarse(idx))
    ax.grid()
plt.show()

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
