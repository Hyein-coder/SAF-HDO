#%%
import pandas as pd

file_res = r"D:\SAF_Nurul\20260330_NewSim\SAF_Sensitivity_Analysis_260401_mjsp_detail.xlsx"
xl = pd.read_excel(file_res, sheet_name=None)

#%% TEA
xl['TEA'].columns = ['ComponentGroup', 'Category', 'C2', 'C1']
xl['TEA'] = xl['TEA'][['ComponentGroup', 'Category', 'C1', 'C2']]

equipments = xl['TEA'].query("ComponentGroup == 'Equipment'").copy()
equipments = equipments[equipments['Category'] != 'total equipment cost']
equipments = equipments.sort_values(equipments.columns.tolist()[-1], ascending=False)

equip_name_map = {
    'capture': 'Carbon Capture',
    'compressor': 'Compressor',
    'distillation': 'Distillation',
    'heater': 'Heater',
    'reactor': 'Reactor',
    'heatx': 'Heat Exchanger',
    'pump': 'Pump',
    'separator': 'Separator',
    'flash': 'Drum',
    'catalyst': 'Catalyst',
    'dryer': 'Dryer',
    'psa': 'PSA'
}
equipments['Category'] = equipments['Category'].replace(equip_name_map)

equip_top_5 = equipments.nlargest(5, 'C2')
equip_others = equipments[~equipments.index.isin(equip_top_5.index)]

equip_others_row = {
    'ComponentGroup': 'Equipment',
    'Category': 'Other Equipments',
}
for j in equip_others.columns.tolist()[2:]:
    equip_others_row[j] = equip_others[j].sum()

equip_others_df = pd.DataFrame([equip_others_row])
equipments_summary = pd.concat([equip_top_5, equip_others_df], ignore_index=True)

capex_others = xl['TEA'].query("ComponentGroup == 'CAPEX'").copy()
capex_others = capex_others[capex_others['Category'] != 'Equipment Cost']
capex_others = capex_others[capex_others['Category'] != 'Total CAPEX']
capex_others = capex_others[capex_others.iloc[:,-1] > 0]
capex_others = capex_others.sort_values(capex_others.columns.tolist()[-1], ascending=False)

capex_name_map = {
    'C_Comp': 'Computers',
    'Site Prep.': 'Site Preparation',
}
capex_others['Category'] = capex_others['Category'].replace(capex_name_map)

capex = pd.concat([equipments_summary, capex_others])

opex = xl['TEA'].query("ComponentGroup == 'OPEX'").copy()
opex = opex[opex['Category'] != 'Total OPEX']
opex = opex.sort_values(opex.columns.tolist()[-1], ascending=False)

opex_name_map = {
    'Catalyst OPEX': 'Catalyst',
    'Capture OPEX': 'Carbon Capture',
    'Property Taxes & Insurance': 'Taxes',
    'Operating Overhead': 'Overhead'
}
opex['Category'] = opex['Category'].replace(opex_name_map)

msp = xl['TEA'].query("ComponentGroup == 'MSP [$/kg]'").copy()
msp['Category'] = msp['ComponentGroup']
msp_breakdown = xl['TEA'].query('ComponentGroup.str.contains("breakdown", case=False, na=False)').copy()
msp_breakdown['Category'] = msp_breakdown['ComponentGroup'].str.replace('in MSP breakdown', '').str.strip()

#%% Numerical values only
def turn_into_numeric(df):
    df.drop(columns='ComponentGroup', inplace=True)
    df.set_index('Category', inplace=True)

turn_into_numeric(capex)
turn_into_numeric(opex)
turn_into_numeric(msp)
turn_into_numeric(msp_breakdown)

total_capex = capex.sum(axis=0)
cf_ratio_capex = -msp_breakdown.loc['CAPEX'] / msp_breakdown.loc['Sales']
capex_msp = capex / total_capex * cf_ratio_capex * msp.iloc[0,:]

total_opex = opex.sum(axis=0)
cf_ratio_opex = msp_breakdown.loc['OPEX'] / msp_breakdown.loc['Sales']
opex_msp = opex / total_opex * cf_ratio_opex * msp.iloc[0,:]

#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

TWOCASE_RC = {
    # --- Figure ---
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.autolayout': False,  # use tight_layout() manually

    # --- Axes ---
    'axes.facecolor': 'white',
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'axes.spines.top': True,  # both spines visible (commented-out removal kept)
    'axes.spines.right': True,
    'axes.labelsize': 12,
    'axes.labelcolor': 'black',
    'axes.titlelocation': 'center',

    # --- Grid ---
    'axes.grid': False,  # no grid in source plot

    # --- Ticks ---
    'xtick.labelsize': 12,
    'xtick.color': 'black',
    'xtick.direction': 'out',
    'ytick.labelsize': 10,
    'ytick.color': 'black',
    'ytick.direction': 'out',

    # --- Font ---
    'font.family': 'sans-serif',
    'font.serif': ['computer modern roman'],
    'font.sans-serif': ['Helvetica Neue LT Pro'],
    'font.size': 10,
    'text.color': 'black',
    'font.weight': '300', 'axes.labelweight': '500',

    # --- Patches (bars, legend swatches) ---
    'patch.edgecolor': '#444444',
    'patch.linewidth': 0.7,
    'patch.force_edgecolor': True,

    # --- Legend ---
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'legend.framealpha': 0.0,  # transparent legend box
    'legend.edgecolor': '#444444',

    # --- Lines ---
    'lines.linewidth': 0.7,

    # --- Saving ---
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}
plt.rcParams.update(TWOCASE_RC)

# CAPEX: cool blue-teal family
capex_colors = [
    '#C4E0D4',  # pale teal (Lightest Green)
    '#B8D4E8',  # light steel blue (Lightest Blue)

    '#CDE3B2',  # soft chartreuse (Green)
    '#90C4D8',  # medium sky blue (Blue)

    '#B4D4CC',  # light sage-teal (Green)
    '#7AB8D0',  # muted cyan (Blue)

    '#AED6C4',  # soft mint (Green)
    '#6AAEC4',  # medium teal-blue (Blue)

    '#A8D8B8',  # soft seafoam (Green)
    '#5898B4',  # deeper blue (Blue)

    '#88C0B8',  # medium aquamarine (Green)
    '#4A88A4',  # slate blue (Blue)

    '#5F8C6E'   # deep juniper green (Darkest Green)
]

# OPEX: warm pink-peach-gold family
opex_colors = [
    '#FEEBD6',  # Very Pale Peach
    '#FFFAF0',  # Warm Cream
    '#FADBC7',  # Desert Sand
    '#FDD280',  # Soft Amber
    '#F5C270',  # Muted Gold
    '#D4A870',  # Warm Tan/Amber
    '#C09763',  # Ochre
    '#E4957D',  # Soft Terracotta
    '#F08080',  # Coral Pink
    '#B56565'   # Deep Coral Rose
]

capex_plot = capex_msp.transpose()
opex_plot = opex_msp.transpose()
scenarios = capex_plot.index.tolist()
scen_labels = scenarios

bar_width = 0.35
gap       = 0.08
x_centers = np.arange(len(scenarios))

capex_pos = x_centers - bar_width/2 - gap/2
opex_pos  = x_centers + bar_width/2 + gap/2

capex_components = capex_plot.columns.tolist()
opex_components  = opex_plot.columns.tolist()

fig, ax = plt.subplots()

for si, scen in enumerate(scenarios):
    # CAPEX stacked bar
    bottom_CAPEX = 0
    for ci, comp in enumerate(capex_components):
        val = capex_plot.loc[scen, comp]
        ax.bar(capex_pos[si], val, bar_width, bottom=bottom_CAPEX,
               color=capex_colors[ci % len(capex_colors)],
               edgecolor='#444444',  # Dark grey border for better segment separation
               )
        bottom_CAPEX += val

    # OPEX stacked bar
    bottom = 0
    for oi, comp in enumerate(opex_components):
        val = opex_plot.loc[scen, comp]
        ax.bar(opex_pos[si], val, bar_width, bottom=bottom,
               color=opex_colors[oi % len(opex_colors)],
               edgecolor='#444444',  # Consistent dark grey border
               )
        bottom += val

    ax.text(x_centers[si], bottom + 0.009, rf'MSP \${bottom + bottom_CAPEX:.2f} kg$^{{\mathbf{{-1}}}}$',
            ha='center', va='bottom', fontsize=9, fontweight='500', color='#444444')
ax.set_xticks(x_centers)
ax.set_xticklabels(scen_labels)
for label in ax.get_xticklabels():
    label.set_fontweight('500')
ax.tick_params(axis='x', pad=10)

for si in range(len(scenarios)):
    ax.text(capex_pos[si]-0.02, -0.025, 'CAPEX',
            ha='center', va='top', fontsize=8, color='#1565C0', fontweight='500')
    ax.text(opex_pos[si]+0.02,  -0.025, 'OPEX',
            ha='center', va='top', fontsize=8, color='#B71C1C', fontweight='500')

ax.set_xlim(-0.6, len(scenarios) - 0.4)
top = max(capex_plot.loc[scenarios].sum(axis=1).max(),
          opex_plot.loc[scenarios].sum(axis=1).max()) * 1.18
ax.set_ylim(0., top)

ax.set_ylabel(r'Minimum Selling Price (\$ kg$^{-1}$)')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

capex_patches = [mpatches.Patch(color=capex_colors[i], label=comp)
                 for i, comp in enumerate(capex_components)]
opex_patches  = [mpatches.Patch(color=opex_colors[i], label=comp)
                 for i, comp in enumerate(opex_components)]

leg1 = ax.legend(handles=capex_patches, title='CAPEX Components', ncol=2,
                 bbox_to_anchor=(1.01, 1), loc='upper left',
                 title_fontproperties={'weight': '500'})
ax.add_artist(leg1)
lef2 = ax.legend(handles=opex_patches, title='OPEX Components', ncol=2,
          bbox_to_anchor=(1.01, 0.28), loc='upper left', borderaxespad=0.01,
          title_fontproperties={'weight': '500'})

for patch in leg1.get_patches() + lef2.get_patches():
    patch.set_edgecolor('#444444')
    patch.set_linewidth(0.7)

plt.tight_layout()
plt.savefig('figures/drHa_TEA_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/drHa_TEA_plot.svg', dpi=150, bbox_inches='tight', format='svg')
print("TEA plot saved.")

#%%
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

# 1. Load Data
df = pd.read_excel('results/TwoCase_LCA.xlsx')
first_col = df.columns[0]
df.set_index(first_col, inplace=True)
df.index = df.index.astype(str).str.strip()

# 2. Filter Scenarios
scenarios   = ['Peak-shifted', 'Heavy-end', 'CAF']
scen_labels = ['Best', 'Worst', 'CAF']
df_plot = df.loc[scenarios].drop(columns=['Emission'], errors='ignore')
components = df_plot.columns.tolist()

# 3. Apply your CAPEX color family to the LCA components
lca_colors = [
    '#B3CDE0',  # pastel blue
    '#CCEBC5',  # pastel mint green
    '#DECBE4',  # pastel lavender
    '#FED9A6',  # pastel orange
    '#F9E79F',  # pastel yellow
    '#B3E2CD',  # pastel aqua/teal
    # '#FDCDAC',  # pastel peach
    '#FBB4AE',  # pastel pink
]

# 4. Setup Plot matching your specified dimensions
fig, ax = plt.subplots(figsize=(6,4))
bar_width = 0.45
x_centers = np.arange(len(scenarios))

# 5. Build Stacked Bars with Borders
tallest_val = 0
for si, scen in enumerate(scenarios):
    bottom = 0
    for ci, comp in enumerate(components):
        val = df_plot.loc[scen, comp]
        if val > 0:
            ax.bar(x_centers[si], val, bar_width, bottom=bottom,
                   color=lca_colors[ci],
                   edgecolor='#444444',  # Dark grey border for segment separation
                   linewidth=0.7)        # Slightly thicker line
            bottom += val

    ax.text(x_centers[si], bottom + 0.5, f'{bottom:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='500', color='#444444')

    if bottom > tallest_val:
        tallest_val = bottom

# 6. Typography and Axis Formatting
ax.set_xticks(x_centers)
ax.set_xticklabels(scen_labels, fontsize=12)
for label in ax.get_xticklabels():
    label.set_fontweight('500')
ax.set_xlim(-0.6, len(scenarios) - 0.4)
ax.set_ylabel(r'GHG Emission (g CO$_2$e MJ$^{-1}$)', fontsize=12)
ax.set_ylim(0, tallest_val*1.1)

# 7. Custom Legend (Outside plot, no frame, bordered patches)
patches = [mpatches.Patch(color=lca_colors[i], label=comp)
           for i, comp in enumerate(components)]

leg = ax.legend(handles=patches, title='LCA Components',
                bbox_to_anchor=(1.01, 1), loc='upper left',
                title_fontproperties={'weight': '500'})

# Apply matching borders to legend
for patch in leg.get_patches():
    patch.set_edgecolor('#444444')
    patch.set_linewidth(0.7)

plt.tight_layout()
plt.savefig('figures/TwoCase_LCA_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/TwoCase_LCA_plot.svg', dpi=150, bbox_inches='tight', format='svg')
print("LCA plot saved.")