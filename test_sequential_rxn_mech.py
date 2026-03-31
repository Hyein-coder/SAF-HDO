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

case_target = "k"
aspen_path = r"D:\SAF_Nurul\20260330_NewSim\260331_new_mechanism_v2_j.bkp"

# sim = AspenSim(aspen_path, case_target=case_target)
sim = AspenSim(aspen_path, case_target=case_target, from_directory=False)

cc = sim.get_carbon_number_composition(sim.prod_stream)
print(cc)
#%%
fig, ax = plt.subplots(1, 1, figsize=(5,4))
plt.plot(cc.keys(), cc.values(), '-', label='Simulation')
sim.set_target('k')
plt.plot(sim.target.keys(), sim.target.values(), 'o', label='k')
sim.set_target('j')
plt.plot(sim.target.keys(), sim.target.values(), '^', label='j')
plt.xlabel("Number of Carbons", fontsize=11)
plt.ylabel("Weight Fraction", fontsize=11)
plt.tight_layout()
plt.legend()
# plt.savefig(os.path.join(os.path.dirname(aspen_path), "fig_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".png"))
plt.show()

# sim.terminate()