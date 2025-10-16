from aspen_utils import AspenSim
import os

aspen_path = r"D:\saf_hdo\aspen\251009_pyrolysis_oil_CC_C6H8O_DGFORM_SMR_solver_R101_normalized_Elemental_properties.bkp"
sim = AspenSim(aspen_path)

pyro = sim.aspen.Tree.FindNode(sim.pyrolyzer_node)
r1 = sim.aspen.Tree.FindNode(sim.rxtor_nodes[0])
#%%
sim.pyrolyzer_num_components = 35
pyro = sim.aspen.Tree.FindNode(sim.pyrolyzer_node)
names_raw = [e.Name.replace(" MIXED", "") for i, e in enumerate(pyro.Elements)]
sim.pyro_comp_names = [names_raw[sim.pyrolyzer_num_components * i] for i in range(sim.pyrolyzer_num_components)]
sim.pyro_yields = [e.Value for i, e in enumerate(pyro.Elements) if i < sim.pyrolyzer_num_components]
# sim.pyro_yields = [e.Value for e in pyro.Elements[:sim.pyrolyzer_num_components]]

#%%
import pandas as pd
file_components = os.path.join(os.getcwd(), r"aspen\251009_pyrolysis_oil_CC_C6H8O_DGFORM_SMR_solver_R101_normalized.xlsx")
with pd.ExcelFile(file_components) as xls:
    df = xls.parse("Sheet2")
df_components = df.loc[:, ["Component ID", "C", "H", "O"]].fillna(0)

#%%
stream_no = sim.pyro_prod_stream
# get carbon, hydrogen, oxygen frac
frac_C = sim.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\STRM_UPP\MASSFRC\MIXED\TOTAL").Value
frac_H = sim.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\STRM_UPP\MASSFRH\MIXED\TOTAL").Value
frac_O = sim.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\STRM_UPP\MASSFRO\MIXED\TOTAL").Value

# get water mass frac
frac_water = sim.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\MASSFRAC\MIXED\WATER").Value

# subtract hydrogen and oxygen frac due to water
frac_H_wo_water = frac_H - frac_water * 2/18
frac_O_wo_water = frac_O - frac_water * 16/18
