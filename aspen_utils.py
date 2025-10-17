import os
import win32com.client as win32
import pythoncom
import time
import pandas as pd
import os
pythoncom.CoInitialize()

data_path = os.path.join(os.getcwd(), "HDO_exp.xlsx")
# data_path = "D:\saf_hdo\HDO_exp.xlsx"
with pd.ExcelFile(data_path) as xls:
    # xls = pd.ExcelFile(data_path)
    data = pd.read_excel(xls).fillna(0)

file_components = os.path.join(os.getcwd(), r"aspen\251009_pyrolysis_oil_CC_C6H8O_DGFORM_SMR_solver_R101_normalized.xlsx")
with pd.ExcelFile(file_components) as xls:
    df = xls.parse("Sheet2")
df_components = df.loc[:, ["Component ID", "C", "H", "O"]].fillna(0)

component_list = [
    'BIOMASS', 'WATER', 'H2', 'O2', 'N2', 'AR', 'CO', 'CO2', 'CH4',
    'C2H4', 'C2H4O', 'C2H4O2', 'C2H6', 'C2H6O', 'C3H6', 'C3H8',
    'C3H8O', 'C3H8O2', 'C4H10', 'C4H4O', 'C4H8O', 'C5H4O2', 'C5H12',
    'C6H6', 'C6H6O', 'C6H8O', 'C6H12', 'C6H12O6', 'C6H14', 'C6H14O6',
    'C7H8', 'C7H8O', 'C7H14', 'C7H14-CY', 'C7H14-ME', 'C8H10', 'C8H10O',
    'C8H16', 'C8H18', 'C8H16-CY', 'C9H10O3', 'C9H18', 'C9H20', 'C9H20-A',
    'C10H22-C', 'C11H14O', 'C11H20-C', 'C11H24', 'C12H16O2', 'C12H20-C',
    'C12H26', 'C13H18O2', 'C13H20O2', 'C13H26', 'C13H28', 'C14H12',
    'C14H20O2', 'C14H28A', 'C14H30', 'C15H28', 'C16H32', 'C16H34',
    'C17H34', 'C17H36', 'C18H38', 'C19H24O2', 'C19H38', 'C19H40', 'C20H40',
    'C20H42', 'C21H24O5', 'C21H42', 'C22H26O5', 'C22H44', 'C23H28O5',
    'C23H46', 'C24H32O3', 'C24H48', 'C25H30O3', 'C25H50', 'C26H42O4',
    'C26H52', 'C27H54', 'C28H56', 'C29H58', 'C30H62', 'SO2', 'NO2',
    'ACIDS', 'ALDEHYDE', 'KETONES', 'ALCOHOLS', 'GUAIACOL', 'LMWS',
    'HMWS', 'EXTRACTI', 'N2COMP', 'SCOMPOUN', 'S', 'C', 'ASH', 'SIO2',
    'LMWLA', 'LMWLB', 'HLB', 'NH3'
]
component_to_carbon_number = {
    "ACIDS": 4,
    "ALDEHYDE": 8,
    "KETONES": 3,
    "ALCOHOLS": 6,
    "GUAIACOL": 7,
    "LMWS": 6,
    "HMWS": 12,
    "EXTRACTI": 20,
    "N2COMP": 8,
    "SCOMPOUN": 12,
    "LMWLA": 16,
    "LMWLB": 12,
    "HLB": 17,
}



class AspenSim(object):
    def __init__(self, aspen_path, case_target='a', visible=True, rxtor_nodes=None):
        self.aspen = win32.Dispatch("Apwn.Document")
        self.aspen_path = aspen_path
        self.aspen.InitFromArchive2(os.path.abspath(aspen_path))
        self.aspen.Visible = visible
        self.aspen.SuppressDialogs = True
        print("Aspen simulation opened. Just wait..")
        time.sleep(10)
        print("Now run the initial simulation..")
        self.aspen.Engine.Run2()
        time.sleep(2)
        print("Initial simulation finished")

        self.prod_stream = "208"
        if rxtor_nodes is None:
            self.rxtor_nodes = [
                "\Data\Blocks\R-201\Input\CONV",
                # "\Data\Blocks\R-202\Input\CONV",
                # "\Data\Blocks\R-203\Input\CONV",
            ]
        else:
            self.rxtor_nodes = rxtor_nodes

        self.data = data
        self.case_target = case_target
        self.target = None
        self.set_target()

        self.component_list = component_list
        self.component_to_carbon_number = component_to_carbon_number
        carbon_number_list = self.data['Carbon'].tolist()
        carbon_number_to_component = {n: [c for c in self.component_list if f"C{n}" in c] for n in carbon_number_list}
        for c, n in self.component_to_carbon_number.items():
            if n in carbon_number_list:
                carbon_number_to_component[n].append(c)
        self.carbon_number_to_component = carbon_number_to_component

        self.n_rxn = []
        self.rxn_indices = []
        self.set_rxtors()

        self.pyrolyzer_node = "\Data\Blocks\R-101\Input\MOLE_YIELD"
        self.pyrolyzer_num_components = 35
        self.pyro_comp_names = []
        self.set_pyrolyzer()
        self.pyro_prod_stream = "146"   #TODO: Should be checked

    def set_target(self, case_target=None):
        if case_target is None:
            case_target = self.case_target

        target_sum = sum(self.data[case_target])
        target_raw = {n: c / target_sum for n, c in zip(self.data['Carbon'], self.data[case_target])}

        target = {n: 0 for n in self.data['Carbon'] if n < 26}
        for n, c in target_raw.items():
            if n < 25:
                target[n] = c
            else:
                target[25] += c

        self.case_target = case_target
        self.target = target
        return case_target, target

    def set_rxtors(self):
        n_rxns = [len(self.aspen.Tree.FindNode(rxtor).Elements) for rxtor in self.rxtor_nodes]
        ri_test = [[i for i in range(n)] for n in n_rxns]

        ri_valid = []
        val_initial = []
        for ii, rii in enumerate(ri_test):
            rxtor = self.aspen.Tree.FindNode(self.rxtor_nodes[ii])
            rii_valid = []
            vii_initial = []
            for n in range(n_rxns[ii]):
                for k in range(10):
                    try:
                        # print(f"Try {rii[n]} for {n}")
                        val0 = rxtor.Elements(f"{rii[n]}").Value
                        rxtor.Elements(f"{rii[n]}").Value = val0 * 0.99
                        rii_valid.append(rii[n])
                        vii_initial.append(val0)
                        break
                    except:
                        rii.append(rii[-1] + 1)
                        rii.remove(rii[n])

            ri_valid.append(rii_valid)
            val_initial.append(vii_initial)

        self.n_rxn = n_rxns
        self.rxn_indices = ri_valid
        self.apply_rxn_coefficients(val_initial)

    def get_carbon_number_composition(self, stream_no):
        s = self.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\MASSFLOW\MIXED")
        massflow = [0 if s.Elements(c).Value is None else s.Elements(c).Value for c in self.component_list]
        hydrocarbon_massflow = {
            n: sum([massflow[self.component_list.index(c)] for c in cs])
            for n, cs in self.carbon_number_to_component.items()
        }
        hydrocarbon_total = sum(hydrocarbon_massflow.values())
        carbon_number_composition = {n: cs / hydrocarbon_total for n, cs in hydrocarbon_massflow.items()}
        return carbon_number_composition

    def get_rxn_coefficients(self):
        rxn_coefficients = []
        for i, ris in enumerate(self.rxn_indices):
            rxn_coef_i = []
            for ii, rxn_idx in enumerate(ris):
                rxtor = self.aspen.Tree.FindNode(self.rxtor_nodes[i])
                rxn_coef_i.append(rxtor.Elements(f"{rxn_idx}").Value)
            rxn_coefficients.append(rxn_coef_i)
        return rxn_coefficients

    def apply_rxn_coefficients(self, rxn_coefficients):
        try:
            for i, ris in enumerate(self.rxn_indices):
                for ii, rxn_idx in enumerate(ris):
                    rxtor = self.aspen.Tree.FindNode(self.rxtor_nodes[i])
                    rxtor.Elements(f"{rxn_idx}").Value = rxn_coefficients[i][ii]
            time.sleep(2)
            print("Running simulation")
            self.aspen.Engine.Run2()
            time.sleep(2)
            print("Simulation finished")
            res_composition = self.get_carbon_number_composition(self.prod_stream)
            return res_composition
        except:
            print("Error in applying rxn coefficients")
            return None

    def get_elemental_composition(self, stream_no):
        frac_C = self.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\STRM_UPP\MASSFRC\MIXED\TOTAL").Value
        frac_H = self.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\STRM_UPP\MASSFRH\MIXED\TOTAL").Value
        frac_O = self.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\STRM_UPP\MASSFRO\MIXED\TOTAL").Value

        # get water mass frac
        frac_water = self.aspen.Tree.FindNode("\Data\Streams\\" + stream_no + "\Output\MASSFRAC\MIXED\WATER").Value

        # subtract hydrogen and oxygen frac due to water
        frac_C_wo_water = frac_C / (1 - frac_water)
        frac_H_wo_water = (frac_H - frac_water * 2 / 18) / (1 - frac_water)
        frac_O_wo_water = (frac_O - frac_water * 16 / 18) / (1 - frac_water)
        elemental_composition = {"C": frac_C_wo_water, "H": frac_H_wo_water, "O": frac_O_wo_water}
        return elemental_composition

    def set_pyrolyzer(self):
        pyro = self.aspen.Tree.FindNode(self.pyrolyzer_node).Elements
        names_raw = [e.Name.replace(" MIXED", "") for i, e in enumerate(pyro)]
        self.pyro_comp_names = [names_raw[self.pyrolyzer_num_components * i] for i in range(self.pyrolyzer_num_components)]

    def get_pyro_yields(self):
        pyro = self.aspen.Tree.FindNode(self.pyrolyzer_node).Elements
        pyro_yields = [e.Value for i, e in enumerate(pyro) if i < self.pyrolyzer_num_components]
        return pyro_yields

    def apply_pyro_yields(self, yields):
        assert len(yields) == self.pyrolyzer_num_components, f"Expected input_yields to have a length of 35, but got {len(yields)}"
        try:
            pyro = self.aspen.Tree.FindNode(self.pyrolyzer_node)
            for i, y in enumerate(yields):
                pyro.Elements[i].Value = y
            time.sleep(2)
            print("Running simulation")
            self.aspen.Engine.Run2()
            time.sleep(2)
            status = self.check_result_status()
            print("Simulation finished: " + status)
            return status
        except:
            print("Error in applying rxn coefficients")
            return None

    def check_result_status(self):
        err_node = self.aspen.Tree.FindNode("\\Data\\Results Summary\\Run-Status\\Output\\PER_ERROR")
        err_msg = "Results Summary status: \n"
        for e in err_node.Elements:
            err_msg = err_msg + "\n" + e.Value

        status = None
        if "errors" in err_msg:
            status = "Error"
        elif "warnings" in err_msg:
            status = "Warning"
        else:
            status = "Converged"
        return status

    def reinit(self):
        self.aspen = win32.Dispatch("Apwn.Document")
        self.aspen.InitFromArchive2(os.path.abspath(self.aspen_path))
        self.aspen.Visible = False
        self.aspen.SuppressDialogs = True

    def terminate(self):
        self.aspen.Close(0)
        pythoncom.CoUninitialize()
