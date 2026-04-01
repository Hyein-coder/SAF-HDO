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

aspen_path = r"D:\SAF_Nurul\20260330_NewSim\260401_pyrolysis_oil_CC_case_k_3.bkp"
case_target = "j"

target_criteria = {
    "k": {"H": 1.8809, "O": 0.0072, "yield": 0, "water": 0},
    "j": {"H": 1.8699, "O": 0.0221, "yield": 0, "water": 0},
    "bench": {"H": 1.9856, "O": 0.0040, "liquid": 84.1/100, "oil": 42.1/100},
}

# sim = AspenSim(aspen_path)
sim = AspenSim(aspen_path, from_directory=False)

def plot_sim_summary():
    cc = sim.get_carbon_number_composition(sim.prod_stream)
    elemental_composition, water_frac = sim.get_elemental_composition_wo_water(sim.prod_stream)
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
    draw_criteria["sim"] = sim_criteria

    fig, axes = plt.subplots(1, 2, figsize=(8,3))
    ax_cc = axes[0]
    ax_cc.plot(cc.keys(), cc.values(), '-', label='Simulation')
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

    width = 0.35/n_case_ar
    ax_ar = axes[1]
    # Plot the bars
    for idx, case in enumerate(cases_ar):
        vals = draw_criteria[case].values()
        ax_ar.bar(x - 0.175 + width * idx, vals, width, label=case)

    # Add labels, title, and custom x-axis tick labels
    ax_ar.set_ylabel('Scores')
    ax_ar.set_xticks(x)
    ax_ar.set_xticklabels(sim_criteria.keys())
    ax_ar.legend()
    plt.tight_layout()

    # plt.savefig(os.path.join(os.path.dirname(aspen_path), "fig_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".png"))
    plt.show()

plot_sim_summary()
# sim.terminate()