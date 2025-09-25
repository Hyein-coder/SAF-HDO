#%% Load packages
import os
folder_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder_dir)

import random
import datetime
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from simul_bayesian_opt_ver250330_jisung_v2 import SimulBayesianOpt

seed = 42
random.seed(seed)
np.random.seed(seed)

#%%
# 냉매 종류 및 Capacity case 별 Initial 값 차이로 인한 구분
CONFIG = "Open&Closed_loop"
CAPACITY_CASE = [3000]
REFRIG = ["NH3"]
N_ITER = 1000
TEMP_SAVE_NAME_COMMON = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

N_SIMUL = 1
PRESSURE_CASE = [7.5]
COMP_TYPE_CASE = ["2-0-3-3"]
NUM_COOLER_CASE = [8]
NUM_VESSEL_CASE = [7]

for i in CAPACITY_CASE:
    globals()[f"X_INIT_CASE_{i}"] = [np.array([[25, 1.5, -5, 28, -10, -10, 50, 20]], dtype=np.float64)]

    globals()[f"BOUNDS_CASE_{i}"] = [np.array([[-20, 1, -20, -20, -20, -20, 30, 15],
                                               [39, 4, 30, 30, 30, 30, 70, 40]], dtype=np.float64)]

TEMP_SAVED_RESULTS_PATH = "temp_saved_results"
TEMP_SAVED_FIG_PATH = "temp_saved_fig"

# %% Define functions
def get_sample(feed_flowrate, configuration, prod_pressure, comp_type, num_coolers, num_vessels, refrig, X_init, bounds):
    Simul = SimulBayesianOpt(feed_flowrate, configuration, prod_pressure, comp_type, num_coolers, num_vessels, refrig)
    Simul.n_iter = N_ITER
    Simul.simul.X_init = X_init
    Simul.simul.bounds = [dict(zip(["name", "type", "domain"], 
                                   [f"x_{i}", "continuous", tuple(bounds[:, i])])) for i in range(bounds.shape[1])]
    
    filename = "{}TPD_{}_{:02d}kPa_comp{}_cooler{}_vessel{}_refrig_{}".format(int(feed_flowrate), 
                                                              configuration, int(prod_pressure*100), 
                                                              comp_type, num_coolers,
                                                              num_vessels, refrig)
    history = Simul.run_bayesian_opt(filename)
    
    X_sample = history["X_sample"]
    Y_sample = history["Y_sample"]
    X_sample_warning = history["X_sample_warning"]
    Y_sample_warning = history["Y_sample_warning"]
    return X_sample, Y_sample, X_sample_warning, Y_sample_warning
        

def draw_fig(result, TPD, config, prod_pressure, comp_type, num_cooler, num_vessel, bounds, *refrig):
    if refrig:
        refrigerant = refrig[0]
    else:
        refrigerant = "no_refrig"
        
    result_Y = result["Y_sample"][:, 0]
    min_val = np.min(result_Y, axis=0)
    idx_val = np.where(result_Y == min_val)[0]
    
    print(f"{TPD}TPD {config} {prod_pressure}bar comp{comp_type} cooler{num_cooler} vessel{num_vessel} refrig{refrig} Result:")
    print(f"Minimum levelized cost ($/tCO2): {min_val}")
    print(f"Minimum levelized cost index: {idx_val} \n")
    
    result_Y_pd = pd.DataFrame(result_Y)
    result_Y_cummin = result_Y_pd.cummin()
    
    # Cumulative minimum
    fig = plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(result_Y_cummin) + 1), result_Y_cummin, color="#ff4d4d", linewidth=2)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Levelized Cost ($/tCO2)", fontsize=12)
    plt.grid(linewidth=0.5)
    plt.title(f"{TPD}TPD {config} Product {prod_pressure}bar w/ COMP{comp_type}, COOLER{num_cooler}, VESSEL{num_vessel} REFRIG{refrigerant} BO result",
              fontsize=15)
    
    fig.savefig(f"Fig_{TPD}TPD_{config}_{prod_pressure}bar_comp{comp_type}_cooler{num_cooler}_vessel{num_vessel}_refrig_{refrigerant}.png")
    plt.close(fig)
    
    # Variation of variable X (L2-norm, normalized by min-max scaling)
    result_X = result["X_sample"]
    X_min = bounds[0, :]
    X_max = bounds[1, :]
    
    X_diff = np.sqrt(np.sum(np.diff((result_X - X_min)/(X_max - X_min), axis=0)**2, axis=1))
    
    fig2 = plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(result_Y_cummin)), X_diff, "o-", color="#5461f0", linewidth=2, markersize=3)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel(r"$|X_{normalized} - X_{normalized,prev}|_2$", fontsize=12)
    plt.grid(linewidth=0.5)
    plt.title(f"{TPD}TPD {config} Product {prod_pressure}bar w/ COMP{comp_type}, COOLER{num_cooler}, VESSEL{num_vessel} REFRIG{refrigerant} BO X L2 norm (normalized)",
              fontsize=15)
    
    fig2.savefig(f"Fig_{TPD}TPD_{config}_{prod_pressure}bar_comp{comp_type}_cooler{num_cooler}_vessel{num_vessel}_refrig_{refrigerant}_L2norm.png")
    plt.close(fig2)
    return fig, fig2
        

# %% Main simulation code
# All simulations are based on carbon clean feed inlet case

if __name__ == "__main__":
    for capacity in CAPACITY_CASE:
        for refrig in REFRIG:
            for i in range(N_SIMUL):
                prod_pressure = PRESSURE_CASE[i]
                comp_type = COMP_TYPE_CASE[i]
                num_cooler = NUM_COOLER_CASE[i]
                num_vessel = NUM_VESSEL_CASE[i]
                X_init = globals()[f"X_INIT_CASE_{capacity}"][i]
                bounds = globals()[f"BOUNDS_CASE_{capacity}"][i]
                
                X_sample, Y_sample, X_sample_warning, Y_sample_warning = \
                    get_sample(capacity, CONFIG, prod_pressure, comp_type, num_cooler, num_vessel, refrig, X_init, bounds)
                    
                result_file = f"temp_saved_results/{capacity}TPD_{CONFIG}_{int(prod_pressure*100):02d}kPa_comp{comp_type}_cooler{num_cooler}_vessel{num_vessel}_refrig_{refrig}.pkl"
                result = pkl.load(open(result_file, "rb"))        
                draw_fig(result, capacity, CONFIG, prod_pressure, comp_type, num_cooler, num_vessel, bounds, refrig)

