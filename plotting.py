import matplotlib.pyplot as plt
import pandas as pd
from aspen_utils import AspenSim
import matplotlib.cm as cm

sim_main = AspenSim(r"D:\saf_hdo\aspen\Basecase_SAF_251128\251127_pyrolysis_oil_CC_case_a_3.bkp")

targets = {}
for a in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
    _, tt = sim_main.set_target(a)
    targets[a] = tt
cmap_target = cm.get_cmap('viridis', len(targets))

#%%
from matplotlib import rcParams
import numpy as np
import os

fs = 10
dpi = 200
config_figure = {'figure.figsize': (7, 3), 'figure.titlesize': fs,
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

#%
dir_all = r"D:\saf_hdo\aspen\grid_20251203_102034_param_0"
param_name = 'Interp1'
df_sa = pd.read_csv(dir_all + r"\results.csv")
df_plot = df_sa.iloc[[0, 10]+[k*10-1 for k in range(2,11)],:]
df_plot = df_plot.sort_index()

N_grid = len(df_plot)
fig, axs = plt.subplots(1, 2, sharey=True)
cmap_rand = cm.get_cmap('RdBu', N_grid)
cmap_target = cm.get_cmap('summer', len(targets))

df_converged = df_plot[df_plot['state'].isin(['Converged', 'Warning'])]
product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

ax = axs[0]
for i, (key, val) in enumerate(targets.items()):
    val_to_draw = {k: v for (k, v) in val.items() if k < 25}
    ax.plot(val_to_draw.keys(), [v * 100 for v in val_to_draw.values()], '-o', markersize=3,
             color=cmap_target(i), alpha=0.5)
for i, (idx, row) in enumerate(df_converged.iterrows()):
    res = row[col_products].tolist()
    val_interp = row[param_name]
    if res is None:
        continue
    ax.plot(product_carbon_range[:-1], [r*100 for r in res[:-1]], '-s', markersize=3,
             color=cmap_rand(i), label=f"{val_interp:.1f}")

ax.set_xlim(5, 25)
x_ticks = np.arange(6, product_carbon_range[-1], 1)
ax.set_xticks(x_ticks)

ax.set_xlabel("Carbon Number")
ax.set_ylabel("Product Distribution (%)")
ax.set_title("(a) Degree of Decomposition")
#%

dir_all = r"D:\saf_hdo\aspen\grid_20251203_105302_param_1"
param_name = 'Interp2'
df_sa = pd.read_csv(dir_all + r"\results.csv")
df_plot = df_sa.iloc[[0]+[k*10-1 for k in range(1,11)]+[80],:]
df_plot = df_plot.sort_index()

N_grid = len(df_plot)
cmap_rand = cm.get_cmap('RdBu', N_grid)
cmap_target = cm.get_cmap('summer', len(targets))

df_converged = df_plot[df_plot['state'].isin(['Converged', 'Warning'])]
product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

ax = axs[1]
for i, (key, val) in enumerate(targets.items()):
    val_to_draw = {k: v for (k, v) in val.items() if k < 25}
    ax.plot(val_to_draw.keys(), [v * 100 for v in val_to_draw.values()], '-o', markersize=3,
             color=cmap_target(i), alpha=0.5)
for i, (idx, row) in enumerate(df_converged.iterrows()):
    res = row[col_products].tolist()
    val_interp = row[param_name]
    if res is None:
        continue
    ax.plot(product_carbon_range[:-1], [r*100 for r in res[:-1]], '-s', markersize=3,
             color=cmap_rand(i), label=f"{val_interp:.1f}")

ax.set_xlim(5, 25)
x_ticks = np.arange(6, product_carbon_range[-1], 1)
ax.set_xticks(x_ticks)

ax.set_xlabel("Carbon Number")
ax.set_title("(b) Shift of Dominant Carbon-chain Length")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(r'D:\saf_hdo\figures\sensitivity_param_combined.png')
plt.show()

#%%
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

read_folder = r"D:\saf_hdo\aspen\sa_20251203"
read_files = os.listdir(os.path.join(read_folder, "converged"))
n_total = len(read_files)
sa_all_rows = []
for idx, f in enumerate(read_files[103:]):
    print(f"Reading file {idx+1}/{n_total}-----")
    sim = AspenSim(os.path.join(read_folder, "converged", f))
    coef = sim.get_rxn_coefficients()
    res = sim.get_carbon_number_composition(sim.prod_stream)
    stat = 'Warning'
    sa_all_rows.append(res2df((0, 0), coef, res, stat))
    sim.aspen.Close()

    df_sa = pd.DataFrame(sa_all_rows)
    df_sa.to_csv(os.path.join(read_folder, "converged_results.csv"))
