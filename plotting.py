import matplotlib.pyplot as plt
import pandas as pd
from aspen_utils import AspenSim
import matplotlib.cm as cm

sim_main = AspenSim(r"D:\saf_hdo\aspen\Simulation_260126\260115_pyrolysis_oil_CC_case_a.bkp")

targets = {}
for a in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
    _, tt = sim_main.set_target(a)
    targets[a] = tt
cmap_target = cm.get_cmap('viridis', len(targets))

#%%
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd

fs = 10
dpi = 200
config_figure = {'figure.figsize': (3.8, 3), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 'font.sans-serif': ['Helvetica Neue LT Pro'],
                 'font.weight': '300', 'axes.titleweight': '400', 'axes.labelweight': '300',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15,
                 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*5, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
rcParams.update(config_figure)

#%
dir_all = r"D:\saf_hdo\aspen\grid_20260129_183257_param_0"
param_name = 'Interp1'
df_sa = pd.read_csv(dir_all + r"\results.csv")
# df_plot = df_sa.iloc[[0, 10]+[k*10-1 for k in range(2,11)],:]
df_plot = df_sa.sort_index()
df_converged = df_plot[df_plot['state'].isin(['Converged', 'Warning'])]

N_grid = len(df_converged)
cmap_rand = cm.get_cmap('RdBu', N_grid)
cmap_target = cm.get_cmap('summer', len(targets))

product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

fig, ax = plt.subplots(1, 1)
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
ax.set_ylim(-0.4, 17.2)
x_ticks = np.arange(6, product_carbon_range[-1], 1)
ax.set_xticks(x_ticks)

ax.set_xlabel("Carbon Number")
ax.set_ylabel("Product Distribution (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1)
plt.tight_layout()
plt.savefig(r'D:\saf_hdo\figures\sensitivity_param_0.png')
plt.savefig(r'D:\saf_hdo\figures\sensitivity_param_0.svg', format='svg', bbox_inches='tight')

plt.show()
#%
dir_all = r"D:\saf_hdo\aspen\grid_20260129_161722_param_1"
param_name = 'Interp2'
df_sa = pd.read_csv(dir_all + r"\results.csv")
# df_plot = df_sa.iloc[[0]+[k*10-1 for k in range(1,11)] + [78],:]
df_plot = df_sa.sort_index()
df_converged = df_plot[df_plot['state'].isin(['Converged', 'Warning'])]

N_grid = len(df_converged)
cmap_rand = cm.get_cmap('RdBu', N_grid)
cmap_target = cm.get_cmap('summer', len(targets))

product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

fig, ax = plt.subplots(1, 1)
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
ax.set_ylim(-0.4, 17.2)
x_ticks = np.arange(6, product_carbon_range[-1], 1)
ax.set_xticks(x_ticks)

ax.set_xlabel("Carbon Number")
ax.set_ylabel("Product Distribution (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1)
plt.tight_layout()
plt.savefig(r'D:\saf_hdo\figures\sensitivity_param_1.png')
plt.savefig(r'D:\saf_hdo\figures\sensitivity_param_1.svg', format='svg', bbox_inches='tight')
plt.show()

#%%
# def res2df(_interp_point, _coef, _res, _stat):
#     row_interp = {}
#     for i, p in enumerate(_interp_point):
#         row_interp[f"Interp{i+1}"] = p
#     row_coef = {}
#     for i, c_rxtor in enumerate(_coef):
#         for idx, c in zip(sim_main.rxn_indices[i], c_rxtor):
#             row_coef[f"R{i+1}_rxn{idx}"] = c
#     row_res = {}
#     for i, r in _res.items():
#         row_res[f"C{i}"] = r
#     row = {**row_interp, **row_coef, **row_res, 'state': _stat}
#     return row
#
# read_folder = r"D:\saf_hdo\aspen\sa_20260129_162406_noise_False"
# read_files = os.listdir(os.path.join(read_folder, "converged"))
# n_total = len(read_files)
# sa_all_rows = []
# for idx, f in enumerate(read_files):
#     print(f"Reading file {idx+1}/{n_total}-----")
#     sim = AspenSim(os.path.join(read_folder, "converged", f))
#     coef = sim.get_rxn_coefficients()
#     res = sim.get_carbon_number_composition(sim.prod_stream)
#     stat = 'Warning'
#     sa_all_rows.append(res2df((0, 0), coef, res, stat))
#     sim.aspen.Close()
#
#     df_sa = pd.DataFrame(sa_all_rows)
#     df_sa.to_csv(os.path.join(read_folder, "converged_results.csv"))

#%%
import matplotlib.pyplot as plt
import numpy as np

# 1. Apply your config_figure
fs = 13
dpi = 200
config_figure = {'figure.figsize': (8, 2.5), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 'font.sans-serif': ['Helvetica Neue LT Pro'],
                 'font.weight': '300', 'axes.titleweight': '400', 'axes.labelweight': '300',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15,
                 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*5, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
plt.rcParams.update(config_figure)

cmap_target = cm.get_cmap('summer', len(targets))
x_subset = [19, 20, 21, 22, 23, 24]
case_keys = list(targets.keys())  # 'a', 'b', etc.
n_cases = len(case_keys)

x = np.arange(len(x_subset))
width = 0.8 / n_cases

fig, ax = plt.subplots()

# 3. Plotting loop
for i, case in enumerate(case_keys):
    y_values = [targets[case][val]*100 for val in x_subset]

    # Position each bar in the group
    offset = (i - n_cases / 2) * width + width / 2

    ax.bar(x + offset, y_values, width,
           label=f'{case}',
           color=cmap_target(i), alpha=0.8)

# 4. Final Polish
ax.set_xticks(x)
ax.set_xticklabels(x_subset)
ax.set_xlabel('Carbon Number', fontweight='300')
ax.set_ylabel('Product Distribution (%)', fontweight='300')

# Personal text box with your font settings
# ax.text(0.05, -0.18, "Distribution: Values 19-25", transform=ax.transAxes,
#         fontsize=fs, ha='left', fontweight='bold', family='sans-serif')

# ax.legend(ncol=4, loc='upper right', frameon=False)
plt.tight_layout()
plt.show()