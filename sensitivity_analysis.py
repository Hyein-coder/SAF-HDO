from aspen_utils import AspenSim
import os
import matplotlib.pyplot as plt
import datetime

save_dir = os.path.join(os.getcwd(), 'results')
simA = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_a_3.bkp")
simI = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_i.bkp", case_target="i")
# simK = AspenSim(r"D:\saf_hdo\aspen\251023_pyrolysis_oil_CC_case_k_rxn_enabled.bkp", case_target="k")
simF = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_f.bkp", case_target="f")

sims = [simA, simI, simF]
original_coefs = []
for s in sims:
    original_coefs.append(s.get_rxn_coefficients())

#%
sim_main = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_a_3.bkp")
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

#%%
print("=== Check If Original Coefficients Match the Data ===")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
dir_all = os.path.join(r"D:\saf_hdo\aspen", f'sa_{timestamp}_basecase')
os.makedirs(dir_all, exist_ok=True)

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
    sim_main.save_simulation_as(os.path.join(dir_all, f'basecase_{n}.bkp'))

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

#%%
from matplotlib import rcParams
import numpy as np
fs = 10
dpi = 200
config_figure = {'figure.figsize': (4, 2.5), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 'font.sans-serif': ['Helvetica Neue LT Pro'],
                 'font.weight': '300', 'axes.titleweight': '400', 'axes.labelweight': '300',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15, 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*5, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
rcParams.update(config_figure)

target_to_draw = ['a', 'f', 'i']
label_to_draw = ['Base', 'Peak-shifted', 'Heavy-end']
for t, l in zip(target_to_draw, label_to_draw):
    fig, ax = plt.subplots(1, 1)
    target_val = {k: v for (k,v) in targets[t].items() if k < 25}
    c = original_coefficients[names.index(t)]
    hdo_res = original_results[names.index(t)]
    hdo_res_draw = {k: v for (k, v) in hdo_res.items() if k < 25}
    hdo_res_cum = np.cumsum([v for v in hdo_res_draw.values()])

    ax2 = ax.twinx()
    ax2.plot(hdo_res_draw.keys(), [v*100 for v in hdo_res_cum], '-', linewidth=2, label="Cumulative",
             color='#e95924', alpha=0.5)
    ax2.set_ylim([-5, 105])
    ax2.tick_params(axis='y', labelcolor='#e95924', color='#e95924')
    ax2.spines['right'].set_color("#e95924")
    ax2.set_ylabel("Cumulative Distribution (%)", color='#e95924')

    ax.plot(target_val.keys(), [v * 100 for v in target_val.values()], '-o', markersize=3, label="Experiment",
            color='k', alpha=0.7)
    ax.plot(hdo_res_draw.keys(), [v * 100 for v in hdo_res_draw.values()], '-s', markersize=3, label="Simulation",
            color='g', alpha=0.7)

    ax.set_xlim([5, 25])
    x_ticks = np.arange(6, 25, 3)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Carbon Number")
    ax.set_ylim([-0.4, 18.2])
    ax.set_ylabel("Product Distribution (%)")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax.legend(lines, labels, loc='center right')

    fig.tight_layout()
    plt.savefig(os.path.join(r'D:\saf_hdo\figures', f'experimental_validation_{l}.png'))
    plt.show()

#%%
from matplotlib import rcParams
import numpy as np
fs = 10
dpi = 200
config_figure = {'figure.figsize': (4, 2.5), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 'font.sans-serif': ['Helvetica Neue LT Pro'],
                 'font.weight': '300', 'axes.titleweight': '400', 'axes.labelweight': '300',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15, 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*5, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
rcParams.update(config_figure)

fig, ax = plt.subplots(1, 1)
colors = ['#428bca', '#d9534f', '#663399']
cmap_target = cm.get_cmap('summer', len(targets))
for i, (key, val) in enumerate(targets.items()):
    val_to_draw = {k: v*100 for (k, v) in val.items() if k < 25}
    ax.plot(val_to_draw.keys(), val_to_draw.values(), '-o', markersize=3, color=cmap_target(i),
            alpha=0.3)
for i, n in enumerate(target_to_draw):
    val_to_draw = {k: v*100 for (k, v) in targets[n].items() if k < 25}
    ax.plot(val_to_draw.keys(), val_to_draw.values(), '-s', markersize=3, color=colors[i],
            label=label_to_draw[i], alpha=0.7)
ax.set_xlim(5, 25)
x_ticks = np.arange(6, 25, 1)
ax.set_xticks(x_ticks)
ax.set_xlabel("Carbon Number")
ax.set_ylabel("Product Distribution (%)")

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(r'D:\saf_hdo\figures\experimental_data.png')
plt.show()
#%%
import numpy as np
import pandas as pd
import time
print("=== Effect of Parameters ===")
N_grid = 11
param_idx = 1

grid_values = np.linspace(0, 1, N_grid)
grid_points = [[(x, 0) for x in grid_values], [(0, x) for x in grid_values]]

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
    res_sa, stat = sim_main.apply_rxn_coefficients(coef_sa)

    if stat == 'Error':
        time.sleep(0.5)
        sim_main.aspen.Reinit()
        time.sleep(0.5)
        res_sa, stat = sim_main.apply_rxn_coefficients(coef_sa)

    sa_coefficients.append(coef_sa)
    sa_results.append(res_sa)
    sa_all_rows.append(res2df((x, y), coef_sa, res_sa, stat))
    if res_sa is not None:
        if 'Error' in stat:
            sim_main.save_simulation_as(os.path.join(dir_error, f'caseSA_{idx}.bkp'))
            sim_main.aspen.Reinit()
        else:
            sim_main.save_simulation_as(os.path.join(dir_conv, f'caseSA_{idx}.bkp'))
    df_sa = pd.DataFrame(sa_all_rows)
    df_sa.to_csv(os.path.join(dir_all, 'results.csv'))

#%
import matplotlib.pyplot as plt
from matplotlib import rcParams
fs = 10
dpi = 200
config_figure = {'figure.figsize': (4.5, 3), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 'font.sans-serif': ['Arial'],  # Avenir LT Std, Helvetica Neue LT Pro, Helvetica LT Std, Helvetica Neue LT Pro
                 'font.weight': '300', 'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15, 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*5, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
rcParams.update(config_figure)

fig, ax = plt.subplots(1, 1)
cmap_rand = cm.get_cmap('magma', N_grid+3)
cmap_original = cm.get_cmap('viridis', len(original_points))

df_sa_converged = df_sa[df_sa['state'].isin(['Converged', 'Warning'])]
product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

for i, (key, val) in enumerate(targets.items()):
    val_to_draw = {k: v for (k, v) in val.items() if k < 25}
    plt.plot(val_to_draw.keys(), [v*100 for v in val_to_draw.values()], '-o', markersize=3, label=key,
             color=cmap_target(i), alpha=0.5)
for i, row in df_sa_converged.iterrows():
    res = row[col_products].tolist()
    if res is None:
        continue
    plt.plot(product_carbon_range[:-1], res[:-1], '-', color=cmap_rand(i))

ax.set_xlim(5, 25)
x_ticks = np.arange(6, product_carbon_range[-1], 1)
ax.set_xticks(x_ticks)

ax.set_xlabel("Carbon Number")
ax.set_ylabel("Product Distribution (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(dir_all, 'plot.png'))
plt.show()

num_fail = df_sa[df_sa['state'] == 'RxnCoefError'].shape[0]
num_error = df_sa[df_sa['state'] == 'Error'].shape[0]
print(f"Simulation failed: {num_fail + num_error} / {N_grid}")
none_indices = [i for i, x in enumerate(sa_results) if x is None]

#%% Draw a heatmap to see the shift of reactions
from utils import plot_list_heatmap

nc_idx_ri_sorted, base_c_length = sim_main.sort_rxn_via_base_components()
sa_coef_sorted = [[sc[0][idx] for nc, idx, ri in nc_idx_ri_sorted[0]] for sc in sa_coefficients]
nc_sorted = [[nc for nc, idx, ri in nc_idx_ri_sorted[0]]]
plot_list_heatmap([sa_coef_sorted], f"Varying Parameter #{param_idx}")
plot_list_heatmap([nc_sorted])

#%% Random data generation
np.random.seed(42)

N = 200
noise = False
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

dir_all = os.path.join(r"D:\saf_hdo\aspen", f'sa_{timestamp}_noise_{noise}')
dir_conv = os.path.join(dir_all, 'converged')
dir_error = os.path.join(dir_all, 'error')
os.makedirs(dir_conv, exist_ok=True)
os.makedirs(dir_error, exist_ok=True)

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
    if idx < 103:
        continue
    print(f"Random SA #{idx}/{N}-----")
    coef_sa = rxn_coef_interp(x, y)
    if noise:
        coef_sa += np.random.normal(0, 0.2, len(coef_sa))
    coef_sa = [[min(max(c, 0), 1) for c in cc] for cc in coef_sa]

    res_sa, stat = sim_main.apply_rxn_coefficients(coef_sa)
    saf_prod = sim_main.aspen.Tree.FindNode(r"\Data\Streams\326\Output\MASSFLMX\MIXED").Value
    print(f"SAF production: {saf_prod:.2f} kg/hr")

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
