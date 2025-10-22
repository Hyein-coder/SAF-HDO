from aspen_utils import AspenSim
import os

simA = AspenSim(r"D:\saf_hdo\aspen\251020_pyrolysis_oil_CC_case_a.bkp")

pyro = simA.aspen.Tree.FindNode(simA.pyrolyzer_node)
r1 = simA.aspen.Tree.FindNode(simA.rxtor_nodes[0])

pyro_elem_comp, pyro_frac_water = simA.get_elemental_composition_wo_water(simA.pyro_prod_stream)
hdo_result = simA.get_carbon_number_composition(simA.prod_stream)
#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
plt.plot(hdo_result.keys(), hdo_result.values(), '-', label='Simulation')
plt.plot(simA.target.keys(), simA.target.values(), 'o', label='Experimental Data')
plt.legend()
plt.show()

#%%
simK = AspenSim(r"D:\saf_hdo\aspen\251021_pyrolysis_oil_CC_case_k.bkp", case_target="k")

#%%
hdo_result_k = simK.get_carbon_number_composition(simA.prod_stream)

fig, ax = plt.subplots(1, 1)
plt.plot(hdo_result_k.keys(), hdo_result_k.values(), '-', label='Simulation')
plt.plot(simK.target.keys(), simK.target.values(), 'o', label='Experimental Data')
plt.legend()
plt.show()
#%%
coeff_a = simA.get_rxn_coefficients()
coeff_k = simK.get_rxn_coefficients()

fig, ax = plt.subplots(1, 1)
plt.plot(range(len(coeff_a[0])), coeff_a[0], '-o', label='A')
plt.plot(range(len(coeff_k[0])), coeff_k[0], '-s', label='K')
plt.legend()
plt.show()

#%%
coeff_a_for_k = [coeff_a[0] + [0 for _ in range(len(coeff_k[0]) - len(coeff_a[0]))]]
hdo_result_a_from_k = simK.apply_rxn_coefficients(coeff_a_for_k)

#%% Check if the result is the same when applying the coefficients from simulation case A
# hdo_result_a_from_k = simK.get_carbon_number_composition(simK.prod_stream)

fig, ax = plt.subplots(1, 1)
plt.plot(hdo_result_a_from_k.keys(), hdo_result_a_from_k.values(), '-', label='Simulation')
plt.plot(simA.target.keys(), simA.target.values(), 'o', label='Experimental Data')
plt.legend()
plt.show()
#%%
import numpy as np
pi = 0.5
coeffs_to_interp = [coeff_a_for_k[0], coeff_k[0]]
num_coeffs = len(coeffs_to_interp[0])

cinterp = []
for i in range(num_coeffs):
    cinterp.append(np.interp(pi, [0, 1], [c[i] for c in coeffs_to_interp]))
coeffs = [cinterp]

hdo_interp = simK.apply_rxn_coefficients(coeffs)
fig, ax = plt.subplots(1, 1)
plt.plot(hdo_interp.keys(), hdo_interp.values(), '-', label='Simulation')
plt.plot(simA.target.keys(), simA.target.values(), 'o', label='Case A')
plt.plot(simK.target.keys(), simK.target.values(), 's', label='Case K')
plt.legend()
plt.show()
#%%

def rxn_coefficients_interp(pi):
    coeffs_to_interp = [coeff_a_for_k[0], coeff_k[0]]
    num_coeffs = len(coeffs_to_interp[0])

    interpolated_coeffs = []
    for i in range(num_coeffs):
        interpolated_coeffs.append(np.interp(pi, [0, 1], [c[i] for c in coeffs_to_interp]))

    rxn_coeffs = [interpolated_coeffs]
    return rxn_coeffs

points_to_interp = np.linspace(0, 1, 5)
res_interp = []
for pi in points_to_interp:
    hdo_interp = simK.apply_rxn_coefficients(rxn_coefficients_interp(pi))
    res_interp.append(hdo_interp)

#%% Try to match case J with extrapolation
from scipy.interpolate import interp1d

def rxn_coefficients_linear_interp(pi):
    coeffs_to_interp = [coeff_a_for_k[0], coeff_k[0]]
    num_coeffs = len(coeffs_to_interp[0])

    interpolated_coeffs = []
    for i in range(num_coeffs):
        linear_interp = interp1d([0, 1], [c[i] for c in coeffs_to_interp],
                                 bounds_error=False, fill_value="extrapolate")
        interpolated_coeffs.append(min(max(float(linear_interp(pi)), 0.), 1.))

    rxn_coeffs = [interpolated_coeffs]
    return rxn_coeffs

pi = 1.2
points_to_interp_long = np.append(points_to_interp, pi)
coeffs_extrap = rxn_coefficients_linear_interp(pi)
hdo_interp = simK.apply_rxn_coefficients(coeffs_extrap)
res_interp_long = res_interp + [hdo_interp]

#%%
import matplotlib.cm as cm
points_draw = points_to_interp_long
res_draw = res_interp_long

target = {}
for a in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
    _, tt = simK.set_target(a)
    target[a] = tt

cmap_interp = cm.get_cmap('bone', len(points_draw)+1)
cmap_target = cm.get_cmap('viridis', len(target))

fig, ax = plt.subplots(1, 1)
for i, pi in enumerate(points_draw):
    plt.plot(res_draw[i].keys(), res_draw[i].values(), '-', label=f'pi={pi}',
             color=cmap_interp(i))
for i, (key, val) in enumerate(target.items()):
    plt.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
             color=cmap_target(i))
plt.legend()
plt.show()


#%% Linearly decreasing a and k coefficients
def plot_res_series(res_plot, points_plot, target_plot=target, save_heading=None):
    cmap_res = cm.get_cmap('bone', len(points_plot) + 1)
    cmap_target = cm.get_cmap('viridis', len(target_plot))

    fig, ax = plt.subplots(1, 1)
    for i, pi in enumerate(points_plot):
        plt.plot(res_plot[i].keys(), res_plot[i].values(), '-', label=f'pi={pi}',
                 color=cmap_res(i))
    for i, (key, val) in enumerate(target_plot.items()):
        plt.plot(val.keys(), val.values(), 'o', markersize=3, label=key,
                 color=cmap_target(i))
    plt.legend()
    if save_heading is not None:
        plt.savefig(f"{save_heading}.png")
    plt.show()

res_reducing_a = []
for pi in points_to_interp:
    coeffs = [[min(cc * pi, 1) for cc in c] for c in coeff_a_for_k]
    hdo_reduced = simK.apply_rxn_coefficients(coeffs)
    status = simK.check_result_status()
    res_reducing_a.append(hdo_reduced)

plot_res_series(res_reducing_a, points_to_interp)

#%%
res_reducing_k = []
for pi in points_to_interp_long:
    coeffs = [[min(cc * pi, 1) for cc in c] for c in coeff_k]
    hdo_reduced = simK.apply_rxn_coefficients(coeffs)
    res_reducing_k.append(hdo_reduced)

plot_res_series(res_reducing_k, points_to_interp_long)

#%%, 'c', 'f', 'g'
target_interp = {c: target[c] for c in ['a', 'b', 'c', 'k']}
plot_res_series(res_interp_long, points_to_interp_long, target_interp)
#%%
target_a = {c: target[c] for c in ['a', 'f', 'h', 'i', 'j']}
plot_res_series(res_reducing_a[1:], points_to_interp[1:], target_a)
#%%
target_k = {c: target[c] for c in ['d', 'e', 'g', 'k']}
plot_res_series(res_reducing_k[3:], points_to_interp_long[3:], target_k)

#%% Add noise to reproduce the exceptional cases
np.random.seed(42)

low = 0
high = 1

def rxn_coefficients_elemental_interp(pi_list):
    coeffs_to_interp = [coeff_a_for_k[0], coeff_k[0]]
    num_coeffs = len(coeffs_to_interp[0])

    interpolated_coeffs = []
    for i in range(num_coeffs):
        interpolated_coeffs.append(np.interp(pi_list[i], [0, 1], [c[i] for c in coeffs_to_interp]))

    rxn_coeffs = [interpolated_coeffs]
    return rxn_coeffs

num_trial = 10
noise_list = []
noise_res = []
for iter in range(num_trial):
    pi_list = np.random.uniform(low=low, high=high, size=len(coeff_k[0])).tolist()
    coeffs = rxn_coefficients_elemental_interp(pi_list)
    hdo_interp = simK.apply_rxn_coefficients(coeffs)
    noise_res.append(hdo_interp)

target_exception = {c: target[c] for c in ['c', 'f', 'g']}
plot_res_series(noise_res, range(num_trial), target_exception)

#%%
def calc_mae(tt, hdo_res):
    err = 0.
    for key, val in tt.items():
        err += abs(hdo_res[key] - val)
    err = err / len(tt)
    return err

res_all = {
    "ak": (res_interp_long, points_to_interp_long),
    "a": (res_reducing_a, points_to_interp),
    "k": (res_reducing_k, points_to_interp_long),
    # "ak_noise": (noise_res, range(num_trial)),
}
def get_min_mae(target_i, res_all):
    min_mse = 1e10
    min_idx = None
    min_res = None
    for key, (res, points) in res_all.items():
        for i, res_i in enumerate(res):
            mse_i = calc_mae(target_i, res_i)
            if mse_i < min_mse:
                min_mse = mse_i
                min_idx = (key, i)
                min_res = res_i
    return min_mse, min_idx, min_res

res_min_mae = {}

for target_name, target_data in target.items():
    min_mae, min_idx, min_res = get_min_mae(target_data, res_all)
    print(f"Min mae for {target_name}: {min_mae} with {min_idx}")

    k, i = min_idx
    res = res_all[k][0][i]
    plot_res_series([res], [f"({k}, {i})"], {target_name: target_data},
                    save_heading=os.path.join(os.getcwd(), "results", f"min_mae_{target_name}_{k}_{i}"))

    res_min_mae[target_name] = {"MAE": min_mae, "min_idx": min_idx, "min_res": min_res}
average_mae = sum([res_tt["MAE"] for res_tt in res_min_mae.values()])/len(target)
print(f"\n Average MAE: {average_mae}")

"""
Min mae for a: 0.002505843388259381 with ('ak', 0)
Min mae for b: 0.012809453855033159 with ('ak', 0)
Min mae for c: 0.010277377280008233 with ('ak', 0)
Min mae for d: 0.014096723803346123 with ('k', 4)
Min mae for e: 0.014881577966650805 with ('k', 4)
Min mae for f: 0.015351980508727065 with ('ak', 1)
Min mae for g: 0.009476820261689289 with ('k', 4)
Min mae for h: 0.019036901897666995 with ('k', 4)
Min mae for i: 0.02091871434373314 with ('k', 3)
Min mae for j: 0.01903563834981138 with ('k', 3)
Min mae for k: 0.004670099647673645 with ('k', 4)

Average MAE: 0.013005557391145383

"""