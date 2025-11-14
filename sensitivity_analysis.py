from aspen_utils import AspenSim
import os
import matplotlib.pyplot as plt

save_dir = os.path.join(os.getcwd(), 'results')
simA = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_a_rxn_index.bkp")
simI = AspenSim(r"D:\saf_hdo\aspen\251111_pyrolysis_oil_CC_case_i_indexing.bkp", case_target="i")
simK = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_k_rxn_enabled.bkp", case_target="k")

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

fig, axs = plt.subplots(1, 2, figsize=(8, 5))
ax_hdo = axs[0]
ax_coef = axs[1]
linestyles = ['-o', '-s', '-^']
for n, c, ls in zip(names, coefs, linestyles):
    hdo_res = sim_main.apply_rxn_coefficients(c)
    ax_hdo.plot(hdo_res.keys(), hdo_res.values(), ls, label=n)
    ax_coef.plot(range(len(c[0])), c[0], ls, label=n)

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

# coarse_test = [0.7]
coarse_test = [0.1, 0.4, 0.7, 1.0]
hdo_vary_x = []
hdo_vary_y = []
sensitivity_dir = os.path.join(r"D:\saf_hdo\aspen", 'sensitivity')
os.makedirs(sensitivity_dir, exist_ok=True)
for i in coarse_test:
    coef_vary_x = rxn_coef_interp(i, 0)
    coef_vary_y = rxn_coef_interp(0, i)

    hdo_vary_x.append(sim_main.apply_rxn_coefficients(coef_vary_x))
    sim_main.save_simulation_as(os.path.join(sensitivity_dir, f'caseI_{i:.1f}.bkp'))

    hdo_vary_y.append(sim_main.apply_rxn_coefficients(coef_vary_y))
    sim_main.save_simulation_as(os.path.join(sensitivity_dir, f'caseK_{i:.1f}.bkp'))

fig, axs = plt.subplots(1, 2, figsize=(8,5))
linestyles = ['-o', '-s', '-^', '-x', '-d']
for ax, hdo in zip(axs, [hdo_vary_x, hdo_vary_y]):
    for d, ls in zip([hdo_orig] + hdo, linestyles):
        ax.plot(d.keys(), d.values(), ls)
    ax.grid()
plt.show()

#%%
coef_save = rxn_coef_interp(0.3, 0.3)
sim_main.apply_rxn_coefficients(coef_save)
# sim_main.save_simulation_as('test_simulation')
initial_path = os.path.abspath(sim_main.aspen_path)
new_path = os.path.join(os.path.dirname(initial_path), "test_simulation" + '.bkp')
sim_main.aspen.SaveAs(new_path)
