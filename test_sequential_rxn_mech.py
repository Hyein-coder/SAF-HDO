#%%
import os
from aspen_utils import AspenSim
from utils import CustomEvaluator
import GPy, GPyOpt
from GPyOpt.methods import ModularBayesianOptimization
import numpy as np
import datetime
import matplotlib
matplotlib.use('TkAgg') # Or 'Agg' for no window at all
import matplotlib.pyplot as plt
import pandas as pd

aspen_path = r"D:\SAF_Nurul\20260330_NewSim\260401_pyrolysis_oil_CC_case_k_6.bkp"
case_target = "k"

target_criteria = {
    "k": {"H": 1.8809, "O": 0.0072, "liquid": 0, "oil": 0},
    "j": {"H": 1.8699, "O": 0.0221, "liquid": 0, "oil": 0},
    "bench": {"H": 1.9856, "O": 0.0040, "liquid": 84.1/100, "oil": 42.1/100},
}

# sim = AspenSim(aspen_path)
sim = AspenSim(aspen_path, from_directory=False)

def plot_sim_summary():
    sim_label = f'sim ({case_target})'

    cc = sim.get_carbon_number_composition(sim.prod_stream)
    elemental_composition = sim.get_elemental_composition("302")
    product = sim.get_mass_flow(sim.prod_stream)
    oil = sim.get_mass_flow("302")
    water = sim.get_mass_flow("301")
    sim_criteria = {
        "H": elemental_composition["H"] / (elemental_composition["C"] / 12),
        "O": (elemental_composition["O"] / 16) / (elemental_composition["C"] / 12),
        "liquid": (oil + water) / product,
        "oil": oil / product,
    }

    draw_criteria = target_criteria.copy()
    draw_criteria[sim_label] = sim_criteria

    fig, axes = plt.subplots(1, 2, figsize=(8,3))
    ax_cc = axes[0]
    ax_cc.plot(cc.keys(), cc.values(), '-', label=sim_label)
    sim.set_target('k')
    ax_cc.plot(sim.target.keys(), sim.target.values(), 'o', label='k')
    sim.set_target('j')
    ax_cc.plot(sim.target.keys(), sim.target.values(), '^', label='j')
    ax_cc.set_xlabel("Number of Carbons", fontsize=11)
    ax_cc.set_ylabel("Weight Fraction", fontsize=11)
    ax_cc.legend()
    plt.tight_layout()

    # Define the x locations for the groups and the width of the bars
    x = np.arange(4)
    cases_ar = draw_criteria.keys()
    n_case_ar = len(cases_ar)

    width = 0.35 / n_case_ar
    ax_ar = axes[1]
    ax_ar_2 = ax_ar.twinx()  # 1. Initialize the secondary y-axis

    # Plot the bars
    key_order = ['liquid', 'oil', 'H', 'O']
    for idx, case in enumerate(cases_ar):
        # 2. Extract primary values ('liquid', 'oil', 'H') and plot on ax_ar
        vals_primary = [draw_criteria[case][e] for e in key_order[:-1]]
        x_primary = x[:3] - 0.175 + width * idx
        ax_ar.bar(x_primary, vals_primary, width, label=case)

        # 3. Extract the secondary value ('O') and plot on ax2
        val_secondary = [draw_criteria[case][key_order[-1]]]
        x_secondary = x[3:] - 0.175 + width * idx
        ax_ar_2.bar(x_secondary, val_secondary, width)
        # Label is omitted here to prevent duplicate entries in the legend

    # Add labels, title, and custom x-axis tick labels
    ax_ar.set_ylabel('liquid, oil, H')
    ax_ar_2.set_ylabel("O")  # 4. Label the new secondary axis

    ax_ar.set_xticks(x)
    ax_ar.set_xticklabels(key_order)

    # Render legend using the primary axis which contains the case labels
    ax_ar.legend()
    plt.tight_layout()

    # plt.savefig(os.path.join(os.path.dirname(aspen_path), "fig_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".png"))
    plt.show()
    return draw_criteria

res = plot_sim_summary()
print(res[f"sim ({case_target})"])
# sim.terminate()

#=======================================================================================================================
"""
Compare the final product yield
"""

sim_k = AspenSim(r"D:\SAF_Nurul\20260330_NewSim\260401_pyrolysis_oil_CC_case_k_6.bkp")
sim_j = AspenSim(r"D:\SAF_Nurul\20260330_NewSim\260401_pyrolysis_oil_CC_case_j_HOadjustment3.bkp")

prod_compare = {
    "k": {'sim': sim_k},
    "j": {'sim': sim_j}
}
for case_name, case_item in prod_compare.items():
    case_sim = case_item['sim']
    prod_compare[case_name]["oil"] = case_sim.get_mass_flow("302")
    prod_compare[case_name]["water"] = case_sim.get_mass_flow("301")
    prod_compare[case_name]["saf"] = case_sim.get_mass_flow("326")
