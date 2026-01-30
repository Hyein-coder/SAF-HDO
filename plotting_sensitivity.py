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
dir_save = r"D:\saf_hdo\results\sensitivity_20260130"
file_simulation = [
    r"D:\saf_hdo\results\sensitivity_20260130\results_random.csv",
    r"D:\saf_hdo\results\sensitivity_20260130\results_param_0.csv",
    r"D:\saf_hdo\results\sensitivity_20260130\results_param_1.csv"
]
file_res = [
    r"D:\saf_hdo\results\sensitivity_20260130\SAF_Sensitivity_Analysis_260130_converged.xlsx",
    r"D:\saf_hdo\results\sensitivity_20260130\SAF_Sensitivity_Analysis_260129_param_0.xlsx",
    r"D:\saf_hdo\results\sensitivity_20260130\SAF_Sensitivity_Analysis_260129_param_1.xlsx"
]
file_type = ['RD', 'PP', 'PH']

dsim, dtea, dlca = [], [], []
for fsim, fres, ftype in zip(file_simulation, file_res, file_type):
    ds = pd.read_csv(fsim)
    ds['case_name'] =  [ftype + str(i) for i in ds['case_num'].tolist()]
    ds = ds.set_index('case_name')

    dt = pd.read_excel(fres, sheet_name="TEA").transpose()
    tags_tea = ['Case Number']
    for i in dt.index[1:].tolist():
        tag = extract_tag(i)
        if tag is None:
            tags_tea.append(np.nan)
        else:
            tags_tea.append(ftype + str(tag))
    # tags_tea = [extract_tag(i) for i in dt.index]
    # tags_tea[0] = 'Case Number'
    tags_row_tea = pd.DataFrame([tags_tea], columns=dt.index, index=['case_num']).transpose()
    dt = pd.concat([tags_row_tea, dt], axis=1)
    new_columns = pd.MultiIndex.from_arrays([dt.loc['ComponentGroup'], dt.loc['Category']],
                                            names=['ComponentGroup', 'Category'])
    dt_multi = dt.copy()
    dt_multi.columns = new_columns
    dt_multi = dt_multi.drop(['ComponentGroup', 'Category'])

    dl = pd.read_excel(fres, sheet_name="LCA")
    tags_lca = [ftype + str(extract_tag(case[:-4])) for case in dl['Case']]
    tags_row_lca = pd.DataFrame([tags_lca], columns=dl.index, index=['case_num']).transpose()
    dl = pd.concat([tags_row_lca, dl], axis=1)

    dsim.append(ds)
    dtea.append(dt_multi)
    dlca.append(dl)

df_sim = pd.concat(dsim)
df_tea = pd.concat(dtea)
df_lca = pd.concat(dlca)

# df_sim = pd.read_csv(file_simulation).set_index('case_num')
# df_raw_tea = pd.read_excel(file_res, sheet_name="TEA").transpose()
# df_raw_lca = pd.read_excel(file_res, sheet_name="LCA")

# tags_lca = [extract_tag(case[:-4]) for case in df_raw_lca['Case']]
# tags_row_lca = pd.DataFrame([tags_lca], columns=df_raw_lca.index, index=['case_num']).transpose()
# df_lca = pd.concat([tags_row_lca, df_raw_lca], axis=1)
# %
# new_columns = pd.MultiIndex.from_arrays([df_tea.loc['ComponentGroup'], df_tea.loc['Category']],
#                                         names=['ComponentGroup', 'Category'])
# df_tea_multi = df_tea.copy()
# df_tea_multi.columns = new_columns

msp = pd.DataFrame(df_tea['MSP [$/kg]'][np.nan].copy().tolist(),
                   index=df_tea['Case Number'][np.nan].copy().tolist())
h2 = pd.DataFrame(df_tea['H2 consumption [kg/hr]'][np.nan].copy().tolist(),
                   index=df_tea['Case Number'][np.nan].copy().tolist())/1000
ng = pd.DataFrame(df_tea['NG feed [kg/hr]'][np.nan].copy().tolist(),
                   index=df_tea['Case Number'][np.nan].copy().tolist())/1000
ncg = pd.DataFrame((df_tea['LG in 416 [kg/hr]'][np.nan].copy() +
                   df_tea['CH4 in 416 [kg/hr]'][np.nan].copy()).tolist(),
                   index=df_tea['Case Number'][np.nan].copy().tolist())/1000
saf = pd.DataFrame(df_tea['SAF production [kg/hr]'][np.nan].copy().tolist(),
                   index=df_tea['Case Number'][np.nan].copy().tolist())/1000
capex = pd.DataFrame(df_tea['CAPEX']['Total CAPEX'].copy().tolist(),
                   index=df_tea['Case Number'][np.nan].copy().tolist())/1e6
opex = pd.DataFrame(df_tea['OPEX']['Total OPEX'].copy().tolist(),
                   index=df_tea['Case Number'][np.nan].copy().tolist())/1e6
lca = pd.concat([df_lca['Alloc_SAF_5'], df_lca['case_num']], axis=1).set_index('case_num')*1000
valid_indices = [i for i in lca.index.tolist()]
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
plt.savefig(os.path.join(dir_save, f"random_line_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
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
plt.savefig(os.path.join(dir_save, f"random_heat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
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
    sc1 = ax.scatter(tx, ty, c=data, cmap=cmap, edgecolors='w', s=100)
    # ax1.set_title('Plot 1: Uniform Dist.\n(Color = A)', fontsize=14)
    # Add colorbar for ax1
    cb1 = plt.colorbar(sc1, ax=ax, shrink=0.6)
    cb1.set_label(label)

plt.tight_layout()
plt.savefig(os.path.join(dir_save, f"random_ternary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
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
plt.savefig(os.path.join(dir_save, f"random_line_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
plt.show()

#%%
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os
import datetime
from matplotlib import rcParams

fs = 14
dpi = 200
config_figure = {'figure.figsize': (14, 4),
                 'figure.titlesize': fs,
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
#TODO: REDRAW
input_data = [sim_selected['param_0'], sim_selected['param_1']]
input_data.append(1 - input_data[0] - input_data[1])
input_label = [pretty_names[0], pretty_names[1], 'Base']
output_data = [h2.values[:,0], msp.iloc[:,0], lca.iloc[:,0]]
output_label = [r"H2 Consumption [$t \cdot h^{-1}$]",
                "MSP [$ per kg SAF]", r"GWP [gCO$_2$e per MJ SAF]"]
output_cmap = ['viridis', 'magma_r', 'magma_r']

def ternary_to_cartesian(a, b, c):
    total = a + b + c
    a, b, c = a / total, b / total, c / total
    x = 0.5 * (2 * b + c)
    y = 0.5 * np.sqrt(3) * c
    return x, y


def draw_ternary_frame(ax):
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], 'k-', lw=2)
    # Labels
    ax.text(0.05, -0.07, input_label[0], fontsize=fs, ha='right', fontweight='300', family='sans-serif')
    ax.text(0.95, -0.07, input_label[1], fontsize=fs, ha='left', fontweight='300', family='sans-serif')
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, input_label[2], fontsize=fs, ha='center', fontweight='300', family='sans-serif')
    ax.axis('off')
    ax.axis('equal')


# Setup Plot
fig, axes = plt.subplots(1, len(output_data))

# Calculate Cartesian coordinates for all points
# Ensure inputs are numpy arrays for stability
tx, ty = ternary_to_cartesian(np.array(input_data[0]),
                              np.array(input_data[1]),
                              np.array(input_data[2]))

for ax, data, label, cmap in zip(axes, output_data, output_label, output_cmap):
    draw_ternary_frame(ax)

    # 1. Create Filled Contours (The main heat map)
    # levels=14 determines how smooth the gradients look.
    # Use extend='both' to capture min/max outliers nicely.
    contour_filled = ax.tricontourf(tx, ty, data, levels=14, cmap=cmap, extend='both')

    # 2. (Optional) Create Contour Lines
    # This draws thin lines between color changes to define boundaries clearly
    contour_lines = ax.tricontour(tx, ty, data, levels=14, colors=['k'],
                                  linewidths=0.5, alpha=0.3)

    # 3. Add Colorbar
    # Note: We pass 'contour_filled' to the colorbar, not a scatter object
    cb = plt.colorbar(contour_filled, ax=ax, shrink=0.6, pad=0.05)
    cb.set_label(label, fontsize=fs-1)
    cb.ax.tick_params(labelsize=fs-2)

plt.tight_layout()
plt.savefig(os.path.join(dir_save, f"random_contour_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
plt.show()