#%%
import os
from aspen_utils import AspenSim
from utils import CustomEvaluator
import GPy, GPyOpt
from GPyOpt.methods import ModularBayesianOptimization
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd

case_target = "k"
# aspen_path = "D:\\SAF-HDO py\\250926_pyrolysis_oil_CC_2.apw"
aspen_path = r"D:\SAF_Nurul\20260330_NewSim\260115_pyrolysis_oil_CC_case_a_wo2ndRxn.apw"


sim = AspenSim(aspen_path, case_target=case_target) # , case_target='j'

cc = sim.get_carbon_number_composition(sim.prod_stream)
print(cc)
#%%
fig, ax = plt.subplots(1, 1, figsize=(5,4))
plt.plot(cc.keys(), cc.values(), '-', label='Simulation')
plt.plot(sim.target.keys(), sim.target.values(), 'o', label='k')
sim.set_target('j')
plt.plot(sim.target.keys(), sim.target.values(), '^', label='j')
plt.xlabel("Number of Carbons", fontsize=11)
plt.ylabel("Weight Fraction", fontsize=11)
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig(os.path.join(os.path.dirname(aspen_path), "fig_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".png"))

sim.terminate()