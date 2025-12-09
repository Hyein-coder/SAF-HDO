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

#%
import numpy as np
import os
dir_all = r"D:\saf_hdo\aspen\grid_20251203_105302_param_1"
param_name = 'Interp2'
df_sa = pd.read_csv(dir_all + r"\results.csv")
df_plot = df_sa.iloc[[0]+[k*10-1 for k in range(1,11)],:]
df_plot = df_plot.sort_index()

N_grid = len(df_plot)
fig, ax = plt.subplots(1, 1)
cmap_rand = cm.get_cmap('RdBu', N_grid)
cmap_target = cm.get_cmap('summer', len(targets))

df_converged = df_plot[df_plot['state'].isin(['Converged', 'Warning'])]
product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

for i, (key, val) in enumerate(targets.items()):
    val_to_draw = {k: v for (k, v) in val.items() if k < 25}
    plt.plot(val_to_draw.keys(), [v * 100 for v in val_to_draw.values()], '-o', markersize=3,
             color=cmap_target(i), alpha=0.5)
for i, (idx, row) in enumerate(df_converged.iterrows()):
    res = row[col_products].tolist()
    val_interp = row[param_name]
    if res is None:
        continue
    plt.plot(product_carbon_range[:-1], [r*100 for r in res[:-1]], '-s', markersize=3,
             color=cmap_rand(i), label=f"{val_interp:.1f}")

ax.set_xlim(5, 25)
x_ticks = np.arange(6, product_carbon_range[-1], 1)
ax.set_xticks(x_ticks)

ax.set_xlabel("Carbon Number")
ax.set_ylabel("Product Distribution (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(dir_all, 'plot_new.png'))
plt.savefig(r'D:\saf_hdo\figures\sensitivity_param_1.png')
plt.show()

#%%
from matplotlib import rcParams
import numpy as np
import os

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

dir_all = r"D:\saf_hdo\aspen\grid_20251203_102034_param_0"
param_name = 'Interp1'
df_sa = pd.read_csv(dir_all + r"\results.csv")
df_plot = df_sa.iloc[[0, 10]+[k*10-1 for k in range(2,11)],:]
df_plot = df_plot.sort_index()

N_grid = len(df_plot)
fig, ax = plt.subplots(1, 1)
cmap_rand = cm.get_cmap('RdBu', N_grid)
cmap_target = cm.get_cmap('summer', len(targets))

df_converged = df_plot[df_plot['state'].isin(['Converged', 'Warning'])]
product_carbon_range = [c for c in sim_main.carbon_number_to_component.keys()]
col_products = [f"C{i}" for i in product_carbon_range]

for i, (key, val) in enumerate(targets.items()):
    val_to_draw = {k: v for (k, v) in val.items() if k < 25}
    plt.plot(val_to_draw.keys(), [v * 100 for v in val_to_draw.values()], '-o', markersize=3,
             color=cmap_target(i), alpha=0.5)
for i, (idx, row) in enumerate(df_converged.iterrows()):
    res = row[col_products].tolist()
    val_interp = row[param_name]
    if res is None:
        continue
    plt.plot(product_carbon_range[:-1], [r*100 for r in res[:-1]], '-s', markersize=3,
             color=cmap_rand(i), label=f"{val_interp:.1f}")

ax.set_xlim(5, 25)
x_ticks = np.arange(6, product_carbon_range[-1], 1)
ax.set_xticks(x_ticks)

ax.set_xlabel("Carbon Number")
ax.set_ylabel("Product Distribution (%)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(dir_all, 'plot_new.png'))
plt.savefig(r'D:\saf_hdo\figures\sensitivity_param_0.png')
plt.show()

#%%
