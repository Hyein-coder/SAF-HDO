import matplotlib.pyplot as plt

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

#%%
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np

xl = pd.read_excel('results/TwoCase_TEA.xlsx', sheet_name=None)

capex = xl['CAPEX ($MM)'].set_index('Case')
opex  = xl['OPEX ($MMperyear)'].set_index('Case')

scenarios   = ['Base case', 'Heavy-end']
scen_labels = ['Base', 'Heavy-end']

# CAPEX: cool blue-teal family
capex_colors = [
    '#4A88A4',  # slate blue
    '#CDE3B2',  # soft chartreuse (New Green)
    '#7AB8D0',  # muted cyan
    '#AED6C4',  # soft mint
    '#90C4D8',  # medium sky blue
    '#B4D4CC',  # light sage-teal
    '#5898B4',  # deeper blue
    '#88C0B8',  # medium aquamarine
    '#C4E0D4',  # pale teal
    '#6AAEC4',  # medium teal-blue
    '#A8D8B8',  # soft seafoam
    '#5F8C6E',  # deep juniper green (New Green)
    '#B8D4E8',  # light steel blue
]

# OPEX: warm pink-peach-gold family
opex_colors = [
    '#F08080',  # coral pink
    '#FDD280',  #: Soft Amber
    '#FEEBD6',  #: Very Pale Peach (Bottom-most segment)
    '#D4A870',  # warm tan/amber
]

fig, ax = plt.subplots()

bar_width = 0.35
gap       = 0.08
x_centers = np.arange(len(scenarios))

capex_pos = x_centers - bar_width/2 - gap/2
opex_pos  = x_centers + bar_width/2 + gap/2

capex_components = capex.columns.tolist()
opex_components  = opex.columns.tolist()

for si, scen in enumerate(scenarios):
    # CAPEX stacked bar
    bottom_CAPEX = 0
    for ci, comp in enumerate(capex_components):
        val = capex.loc[scen, comp]
        ax.bar(capex_pos[si], val, bar_width, bottom=bottom_CAPEX,
               color=capex_colors[ci % len(capex_colors)],
               edgecolor='#444444',  # Dark grey border for better segment separation
               )
        bottom_CAPEX += val

    # OPEX stacked bar
    bottom = 0
    for oi, comp in enumerate(opex_components):
        val = opex.loc[scen, comp]
        ax.bar(opex_pos[si], val, bar_width, bottom=bottom,
               color=opex_colors[oi % len(opex_colors)],
               edgecolor='#444444',  # Consistent dark grey border
               )
        bottom += val

    ax.text(x_centers[si], bottom + 0.009, f'MSP ${bottom + bottom_CAPEX:.2f}/kg',
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
top = max(capex.loc[scenarios].sum(axis=1).max(),
          opex.loc[scenarios].sum(axis=1).max()) * 1.18
ax.set_ylim(0., top)

ax.set_ylabel(r'Component-wise MSP (\$ kg$^{-1}$ SAF)')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

capex_patches = [mpatches.Patch(color=capex_colors[i], label=comp)
                 for i, comp in enumerate(capex_components)]
opex_patches  = [mpatches.Patch(color=opex_colors[i], label=comp)
                 for i, comp in enumerate(opex_components)]

leg1 = ax.legend(handles=capex_patches, title='CAPEX Components',
                 bbox_to_anchor=(1.01, 1), loc='upper left',
                 title_fontproperties={'weight': '500'})
ax.add_artist(leg1)
lef2 = ax.legend(handles=opex_patches, title='OPEX Components',
          bbox_to_anchor=(1.01, 0.28), loc='upper left', borderaxespad=0.01,
          title_fontproperties={'weight': '500'})

for patch in leg1.get_patches() + lef2.get_patches():
    patch.set_edgecolor('#444444')
    patch.set_linewidth(0.7)

plt.tight_layout()
plt.savefig('figures/TwoCase_TEA_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/TwoCase_TEA_plot.svg', dpi=150, bbox_inches='tight', format='svg')
print("Saved.")