#%%
import os
from aspen_utils import AspenSim

aspen_path = "D:/saf_hdo/aspen/250926_pyrolysis_oil_CC.apw"
sim = AspenSim(aspen_path)

cc = sim.get_carbon_number_composition(sim.prod_stream)
print(cc)
