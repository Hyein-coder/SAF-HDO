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

