from aspen_utils import AspenSim
import os

aspen_path = r"D:\saf_hdo\aspen\251017_Hyein4_c24h32o8.bkp"
sim = AspenSim(aspen_path)

pyro = sim.aspen.Tree.FindNode(sim.pyrolyzer_node)
r1 = sim.aspen.Tree.FindNode(sim.rxtor_nodes[0])

pyro_elem_comp, pyro_frac_water = sim.get_elemental_composition_wo_water(sim.pyro_prod_stream)
#%%
target_pyro = {"C": 0.6017, "H": 0.0602, "O": 0.3367, "N": 0.0013}
target_pyro_water = 0.10

target_hdo_water = 0.32

