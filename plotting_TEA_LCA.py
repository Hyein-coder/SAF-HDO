import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import re
import os
import numpy as np

# Define a function to extract the number from the column name
def extract_tag(name):
    # Regex pattern: look for an underscore followed by digits at the end of the string
    match = re.search(r'_(\d+)$', name)
    if match:
        return int(match.group(1))
    return None # Return None if pattern doesn't match
#%%
args = [
    {"name" : "param_0",
     "file_simulation" : r"D:\SAF_Nurul\Sensitivity_1210\results_param_0.csv",
     "file_res" : r"D:\SAF_Nurul\Sensitivity_1210\SAF_Sensitivity_Analysis_251203_param_0.xlsx",
     "pretty_name": "Heavy-end"
},
    {"name" : "param_1",
     "file_simulation" : r"D:\SAF_Nurul\Sensitivity_1210\results_param_1.csv",
     "file_res" : r"D:\SAF_Nurul\Sensitivity_1210\SAF_Sensitivity_Analysis_251204_param_1.xlsx",
     "pretty_name": "Peak Shift"
}
]

fig_name = os.path.join(r"D:\SAF_Nurul\Sensitivity_1210", "params_all.png")

def get_data(_file_simulation, _file_res):
    df_sim = pd.read_csv(_file_simulation)
    df_raw_tea = pd.read_excel(_file_res, sheet_name="TEA").transpose()
    df_raw_lca = pd.read_excel(_file_res, sheet_name="LCA")

    tags_tea = [extract_tag(col) for col in df_raw_tea.index]
    tags_tea[0] = 'Case Number'
    tags_row_tea = pd.DataFrame([tags_tea], columns=df_raw_tea.index, index=['case_num']).transpose()
    df_tea = pd.concat([tags_row_tea, df_raw_tea], axis=1)

    tags_lca = [extract_tag(case[:-4]) for case in df_raw_lca['Case']]
    tags_row_lca = pd.DataFrame([tags_lca], columns=df_raw_lca.index, index=['case_num']).transpose()
    df_lca = pd.concat([tags_row_lca, df_raw_lca], axis=1)
    #%
    new_columns = pd.MultiIndex.from_arrays([df_tea.loc['ComponentGroup'], df_tea.loc['Category']], names=['ComponentGroup', 'Category'])
    df_tea_multi = df_tea.copy()
    df_tea_multi.columns = new_columns

    msp = pd.DataFrame(df_tea_multi['MSP [$/kg]'][np.nan].iloc[2:].tolist(), index=df_tea_multi['Case Number'][np.nan].iloc[2:].tolist())
    lca = pd.concat([df_lca['Alloc_SAF_5'], df_lca['case_num']], axis=1).set_index('case_num')
    sim_selected = df_sim.iloc[lca.index.tolist()]
    return msp, lca, sim_selected

from matplotlib import rcParams

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

fig, axs = plt.subplots(1, 2, sharey=True)
axs2 = []
for i, arg in enumerate(args):
    idx_param = arg['name']
    msp, lca, sim_selected = get_data(arg['file_simulation'], arg['file_res'])

    ax = axs[i]
    ax.plot(sim_selected[idx_param], msp, '-o', c='blue', markersize=3, linewidth=2, alpha=0.5)
    # ax.tick_params(axis='y', labelcolor='blue', colors='blue')
    # ax.spines['left'].set_color("blue")

    ax2 = ax.twinx()
    ax2.plot(sim_selected[idx_param], lca, '-o', c='orange', markersize=3, linewidth=2, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='orange', colors='orange')
    ax2.spines['right'].set_color("orange")
    ax2.set_ylim([0.011, 0.023])
    axs2.append(ax2)

    ax.set_xlabel(arg['pretty_name'])

axs[0].set_ylabel('MSP [$/kg]')
axs2[0].tick_params(labelright=False)
axs2[1].set_ylabel('GWP [g CO2e/MJ SAF]', color='orange')
plt.tight_layout()
plt.savefig(fig_name)
plt.show()

#%%
file_simulation = r"D:\SAF_Nurul\Sensitivity_1210\results_random_final.csv"
file_res = r"D:\SAF_Nurul\Sensitivity_1210\SAF_Sensitivity_Analysis_251204_converged_corrected.xlsx"

df_sim = pd.read_csv(file_simulation).set_index('case_num')
df_raw_tea = pd.read_excel(file_res, sheet_name="TEA").transpose()
df_raw_lca = pd.read_excel(file_res, sheet_name="LCA")

tags_tea = [extract_tag(col) for col in df_raw_tea.index]
tags_tea[0] = 'Case Number'
tags_row_tea = pd.DataFrame([tags_tea], columns=df_raw_tea.index, index=['case_num']).transpose()
df_tea = pd.concat([tags_row_tea, df_raw_tea], axis=1)

tags_lca = [extract_tag(case[:-4]) for case in df_raw_lca['Case']]
tags_row_lca = pd.DataFrame([tags_lca], columns=df_raw_lca.index, index=['case_num']).transpose()
df_lca = pd.concat([tags_row_lca, df_raw_lca], axis=1)
# %
new_columns = pd.MultiIndex.from_arrays([df_tea.loc['ComponentGroup'], df_tea.loc['Category']],
                                        names=['ComponentGroup', 'Category'])
df_tea_multi = df_tea.copy()
df_tea_multi.columns = new_columns

msp = pd.DataFrame(df_tea_multi['MSP [$/kg]'][np.nan].iloc[2:].copy().tolist(),
                   index=df_tea_multi['Case Number'][np.nan].iloc[2:].copy().tolist())
h2 = pd.DataFrame(df_tea_multi['H2 consumption [kg/hr]'][np.nan].iloc[2:].copy().tolist(),
                   index=df_tea_multi['Case Number'][np.nan].iloc[2:].copy().tolist())/1000
ng = pd.DataFrame(df_tea_multi['NG feed [kg/hr]'][np.nan].iloc[2:].copy().tolist(),
                   index=df_tea_multi['Case Number'][np.nan].iloc[2:].copy().tolist())/1000
ncg = pd.DataFrame((df_tea_multi['LG in 416 [kg/hr]'][np.nan].iloc[2:].copy() +
                   df_tea_multi['CH4 in 416 [kg/hr]'][np.nan].iloc[2:].copy()).tolist(),
                   index=df_tea_multi['Case Number'][np.nan].iloc[2:].copy().tolist())/1000
saf = pd.DataFrame(df_tea_multi['SAF production [kg/hr]'][np.nan].iloc[2:].copy().tolist(),
                   index=df_tea_multi['Case Number'][np.nan].iloc[2:].copy().tolist())/1000
capex = pd.DataFrame(df_tea_multi['CAPEX']['Total CAPEX'].iloc[2:].copy().tolist(),
                   index=df_tea_multi['Case Number'][np.nan].iloc[2:].copy().tolist())/1e6
opex = pd.DataFrame(df_tea_multi['OPEX']['Total OPEX'].iloc[2:].copy().tolist(),
                   index=df_tea_multi['Case Number'][np.nan].iloc[2:].copy().tolist())/1e6
lca = pd.concat([df_lca['Alloc_SAF_5'], df_lca['case_num']], axis=1).set_index('case_num')*1000
valid_indices = [int(i) for i in lca.index.tolist()]
sim_selected = df_sim.loc[valid_indices].copy()

pretty_names = ['Heavy-end', 'Peak Shift']
params = ['param_0', 'param_1']
#%% TRI-AXIS PLOT
from matplotlib import rcParams
import datetime
output_data = [saf, msp, lca]
output_label = ['SAF Production [t/h]', 'MSP [$/kg SAF]', 'GWP [g CO2/MJ SAF']
output_scale = [(np.floor(min(x.iloc[:,0])*0.99), np.ceil(max(x.iloc[:,0])*1.01)) for x in output_data]
# output_scale = [(min(x.iloc[:,0])*0.95, max(x.iloc[:,0])*1.05) for x in output_data]

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

fig, axs = plt.subplots(1, 2, sharey=True)
axs2 = []
axs3 = []
axs4 = []
for i, idx_param in enumerate(params):
    ax = axs[i]
    ax.plot(sim_selected[idx_param], output_data[0], 'o', c='blue', markersize=5, linewidth=2, alpha=0.5)
    # ax.tick_params(axis='y', labelcolor='blue', colors='blue')
    # ax.spines['left'].set_color("blue")

    ax2 = ax.twinx()
    ax2.plot(sim_selected[idx_param], output_data[1], 'o', c='orange', markersize=5, linewidth=2, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='orange', colors='orange')
    ax2.spines['right'].set_color("orange")
    ax2.set_ylim(output_scale[1])
    axs2.append(ax2)

    ax3 = ax.twinx()
    if i < 1:
        ax3.spines["right"].set_position(("outward", 10))
    else:
        ax3.spines["right"].set_position(("outward", 30))
    ax3.plot(sim_selected[idx_param], output_data[2], 'o', c='green', markersize=5, linewidth=2, alpha=0.5)

    ax3.tick_params(axis='y', labelcolor='green', color='green')
    ax3.spines['right'].set_color("green")
    ax3.set_ylim(output_scale[2])
    axs3.append(ax3)

    ax.set_xlabel(pretty_names[i])

axs[0].set_ylabel(output_label[0])
axs2[0].tick_params(labelright=False)
axs2[1].set_ylabel(output_label[1], color='orange')
axs3[0].tick_params(labelright=False)
axs3[1].set_ylabel(output_label[2], color='green')
plt.tight_layout()
plt.savefig(os.path.join(r"D:\SAF_Nurul\Sensitivity_1210", f"random_line_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
plt.show()

#%% PARAMETER SPACE PLOT
from matplotlib import rcParams
import datetime
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

fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
input_data = [sim_selected['param_0'], sim_selected['param_1']]
input_label = [pretty_names[0], pretty_names[1]]
output_data = [ng, msp]
output_label = ['NG Consumption [t/h]', 'MSP [$/kg SAF]']
for ax, z_val, z_name in zip(axs, output_data, output_label):
    # sc = ax.scatter(input_data[0].iloc[:,0], input_data[1].iloc[:,0], c=z_val.iloc[:,0], cmap='viridis', s=50, alpha=0.8)
    sc = ax.scatter(input_data[0], input_data[1], c=z_val.iloc[:,0], cmap='viridis', s=50, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, label=z_name)
    ax.set_xlabel(input_label[0])
    ax.set_ylabel(input_label[1])
# plt.title('Heatmap of Z values at (X, Y) coordinates')
# plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(r"D:\SAF_Nurul\Sensitivity_1210", f"random_heat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
plt.show()

#%% TERNARY PLOT
fs = 10
dpi = 200
config_figure = {'figure.figsize': (14, 4),
                 'figure.titlesize': fs,
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

input_data = [sim_selected['param_0'], sim_selected['param_1']]
input_data.append(1 - input_data[0] - input_data[1])
input_label = [pretty_names[0], pretty_names[1], 'Base']
output_data = [h2, msp.iloc[:,0], lca.iloc[:,0]]
output_label = ["H2 Consumption [t/h]", "MSP [$/kg SAF]", r"GWP [g CO$_2$-eq/kg SAF]"]
output_cmap = ['viridis', 'magma_r', 'magma_r']
# 1. Helper function for coordinates
def ternary_to_cartesian(a, b, c):
    total = a + b + c
    a, b, c = a/total, b/total, c/total
    x = 0.5 * (2 * b + c)
    y = 0.5 * np.sqrt(3) * c
    return x, y

# 2. Helper to draw the triangle frame on a specific axis
def draw_ternary_frame(ax):
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], 'k-', lw=2)
    # Labels
    ax.text(0.05, -0.07, input_label[0], fontsize=12, ha='right', fontweight='bold')
    ax.text(0.95, -0.07, input_label[1], fontsize=12, ha='left', fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, input_label[2], fontsize=12, ha='center', fontweight='bold')
    ax.axis('off')
    ax.axis('equal')

# fig, axes = plt.subplots(2, int(np.ceil(len(output_data)/2)))
# axes = axes.flatten()
fig, axes = plt.subplots(1, len(output_data))

tx, ty = ternary_to_cartesian(*input_data)
for ax, data, label, cmap in zip(axes, output_data, output_label, output_cmap):
    draw_ternary_frame(ax)
    sc1 = ax.scatter(tx, ty, c=data, cmap=cmap, edgecolors='w', s=50)
    # ax1.set_title('Plot 1: Uniform Dist.\n(Color = A)', fontsize=14)
    # Add colorbar for ax1
    cb1 = plt.colorbar(sc1, ax=ax, shrink=0.6)
    cb1.set_label(label)

plt.tight_layout()
plt.savefig(os.path.join(r"D:\SAF_Nurul\Sensitivity_1210", f"random_ternary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
plt.show()

#%% DUAL INPUT - DUAL OUTPUT PLOT
from matplotlib import rcParams
import datetime
input_data = [sim_selected['param_0'], sim_selected['param_1']]
input_label = pretty_names
output_data = [h2, saf, ng]
output_label = [r'H$_2$ Consumption [t/h]', 'SAF Production [t/h]', r'NG Consumption [t/h]']
output_scale = [(np.floor(min(x.iloc[:,0])*0.99), np.ceil(max(x.iloc[:,0])*1.01)) for x in output_data]
# output_scale = [(min(x.iloc[:,0])*0.95, max(x.iloc[:,0])*1.05) for x in output_data]

colormaps = ["#004166", "#e95924", "#198747"]
# colormaps = ["blue", "orange", "green"]

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

fig, axs = plt.subplots(1, 2, sharey=True)
axs2 = []
axs3 = []
axs4 = []
for i, input_i in enumerate(input_data):
    ax = axs[i]
    ax.plot(input_i, output_data[0], 'o', c=colormaps[0], markersize=5, linewidth=2, alpha=0.5)
    # ax.tick_params(axis='y', labelcolor='blue', colors='blue')
    # ax.spines['left'].set_color("blue")

    ax2 = ax.twinx()
    ax2.plot(input_i, output_data[1], 'o', c=colormaps[1], markersize=5, linewidth=2, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=colormaps[1], colors=colormaps[1])
    ax2.spines['right'].set_color(colormaps[1])
    ax2.set_ylim(output_scale[1])
    axs2.append(ax2)

    ax3 = ax.twinx()
    if i < 1:
        ax3.spines["right"].set_position(("outward", 10))
    else:
        ax3.spines["right"].set_position(("outward", 35))
    ax3.plot(input_i, output_data[2], 'o', c=colormaps[2], markersize=5, linewidth=2, alpha=0.5)

    ax3.tick_params(axis='y', labelcolor=colormaps[2], color=colormaps[2])
    ax3.spines['right'].set_color(colormaps[2])
    ax3.set_ylim(output_scale[2])
    axs3.append(ax3)

    ax.set_xlabel(input_label[i])

axs[0].set_ylabel(output_label[0])
axs2[0].tick_params(labelright=False)
axs2[1].set_ylabel(output_label[1], color=colormaps[1])
axs3[0].tick_params(labelright=False)
axs3[1].set_ylabel(output_label[2], color=colormaps[2])
plt.tight_layout()
plt.savefig(os.path.join(r"D:\SAF_Nurul\Sensitivity_1210", f"random_line_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
plt.show()

