import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

xl = pd.read_excel('results/TwoCase_TEA.xlsx', sheet_name=None)

capex = xl['CAPEX ($MM)'].set_index('Case')
opex  = xl['OPEX ($MMperyear)'].set_index('Case')

scenarios   = ['Base case', 'Heavy-end']
scen_labels = ['Base Case', 'Heavy End']

# capex_colors = ['#1565C0','#1E88E5','#42A5F5','#90CAF9','#0D47A1',
#                 '#1976D2','#64B5F6','#BBDEFB','#E3F2FD','#82B1FF',
#                 '#448AFF','#2979FF','#2962FF']
# opex_colors  = ['#B71C1C','#E53935','#EF9A9A','#FF7043']

# CAPEX: cool blue-teal family
# TODO: WITH BORDERS
capex_colors = [
    '#B8D4E8',  # light steel blue
    '#CDE3B2',  # soft chartreuse (New Green)
    '#5F8C6E',  # deep juniper green (New Green)
    '#AED6C4',  # soft mint
    '#90C4D8',  # medium sky blue
    '#B4D4CC',  # light sage-teal
    '#5898B4',  # deeper blue
    '#88C0B8',  # medium aquamarine
    # '#9CCCC0',  # soft aqua
    '#4A88A4',  # slate blue
    '#A8D8B8',  # soft seafoam
    '#7AB8D0',  # muted cyan
    '#C4E0D4',  # pale teal
    '#6AAEC4',  # medium teal-blue
]

# Updated CAPEX: Enhanced contrast with greener accents
# capex_colors = [
#     '#D4ECE8',  # very pale teal (Original)
#     '#B8D4E8',  # light steel blue (Original)
#     '#CDE3B2',  # soft chartreuse (New Green)
#     '#A2C5AC',  # pale fern green (New Green)
#     '#86AF91',  # muted sage green (New Green)
#     '#5F8C6E',  # deep juniper green (New Green)
#     '#9CCCC0',  # soft aqua (Original)
#     '#6AAEC4',  # medium teal-blue (Original)
#     '#4A7C77',  # dark pine teal (New Green)
#     '#90C4D8',  # medium sky blue (Original)
#     '#7AB8D0',  # muted cyan (Original)
#     '#5898B4',  # deeper blue (Original)
#     '#4A88A4',  # slate blue (Original)
# ]

# OPEX: warm pink-peach-gold family
opex_colors = [
    '#F08080',  # coral pink
    '#F4C28A',  # warm peach
    '#E8C84A',  # golden yellow
    '#D4A870',  # warm tan/amber
]

fig, ax = plt.subplots(figsize=(11, 8))

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
               edgecolor='white', linewidth=0.5)
        bottom_CAPEX += val
    # ax.text(capex_pos[si], bottom + 0.003, f'{bottom:.2f}',
    #         ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1565C0')

    # OPEX stacked bar
    bottom = 0
    for oi, comp in enumerate(opex_components):
        val = opex.loc[scen, comp]
        ax.bar(opex_pos[si], val, bar_width, bottom=bottom,
               color=opex_colors[oi % len(opex_colors)],
               edgecolor='white', linewidth=0.5)
        bottom += val
    # ax.text(opex_pos[si], bottom + 0.003, f'{bottom:.2f}',
    #         ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B71C1C')
    ax.text(x_centers[si], bottom + 0.003, f'MSP ${bottom+bottom_CAPEX:.2f}/kg',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#B71C1C')

ax.set_xticks(x_centers)
ax.set_xticklabels(scen_labels, fontsize=12)

for si in range(len(scenarios)):
    ax.text(capex_pos[si], -0.055, 'CAPEX\n($MM)',
            ha='center', va='top', fontsize=8, color='#1565C0', fontweight='bold')
    ax.text(opex_pos[si],  -0.055, 'OPEX\n($MM/yr)',
            ha='center', va='top', fontsize=8, color='#B71C1C', fontweight='bold')

ax.set_xlim(-0.6, len(scenarios) - 0.4)
top = max(capex.loc[scenarios].sum(axis=1).max(),
          opex.loc[scenarios].sum(axis=1).max()) * 1.18
ax.set_ylim(-0.12, top)

ax.set_ylabel('Cost ($MM)', fontsize=12)
ax.set_title('TEA Cost Comparison: CAPEX & OPEX by Scenario\n(stacked by component)',
             fontsize=13, fontweight='bold', pad=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.35)

capex_patches = [mpatches.Patch(color=capex_colors[i], label=comp)
                 for i, comp in enumerate(capex_components)]
opex_patches  = [mpatches.Patch(color=opex_colors[i], label=comp)
                 for i, comp in enumerate(opex_components)]

leg1 = ax.legend(handles=capex_patches, title='CAPEX Components',
                 bbox_to_anchor=(1.01, 1), loc='upper left',
                 fontsize=8, title_fontsize=9, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=opex_patches, title='OPEX Components',
          bbox_to_anchor=(1.01, 0.38), loc='upper left',
          fontsize=8, title_fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('figures/TwoCase_TEA_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/TwoCase_TEA_plot.svg', dpi=150, bbox_inches='tight', format='svg')
print("Saved.")