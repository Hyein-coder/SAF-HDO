import numpy as np


# %% Open-loop configuration
class OpenLoop():
    def __init__(self, feed_flowrate, prod_pressure, comp_type, num_coolers, num_vessels, refrig = None):
        self.feed_flowrate = feed_flowrate
        self.prod_pressure = prod_pressure
        self.comp_type = comp_type
        self.num_coolers = num_coolers
        self.num_vessels = num_vessels
        self.refrig = refrig
        
        self.num_comp_list = list(map(int, self.comp_type.split("-")))
        self.num_compressors = np.sum(np.array(self.num_comp_list))
        
        # Nodes for product pressure
        self.prod_nodes = [f"\\Data\\Blocks\\C-{self.num_comp_list[0]}\\Input\\PRES",
                           f"\\Data\\Blocks\\V-{self.num_comp_list[1]}\\Input\\P_OUT"]
        
        # Node for the feed flow rate
        self.feed_node = ["\\Data\\Streams\\FEED\\Input\\TOTFLOW\\MIXED"]
        
        # Initial value of optimization variable X
        self.X_init = self._get_X_init()
        
        # Bounds of optimization variable X
        self.bounds = self._get_bounds()
        
        # Constraints of optimization variable X
        self.constraints = self._get_constraints()
        
        # Variable names of X
        self.X_names = self._get_X_names()
        
        
    def _get_X_init(self):
        # Number of H-1 hot stream outlet temperature settings
        self.X_hot_streams = 1
        
        # Number of H-1 cold streams outlet temperatures settings
        self.X_cold_streams = self.num_comp_list[1] - 1    # -1 for DoF
        
        # Number of compressors pressure ratios settings
        self.X_compressors = self.num_compressors - 1
        
        # Dimension of X
        self.X_dim = self.X_hot_streams + self.X_cold_streams + self.X_compressors
        
        # Set X_init
        self.X_init = np.empty(shape=(1, self.X_dim))
        self.X_init[:, :self.X_hot_streams] = 36
        self.X_init[:, self.X_hot_streams:self.X_hot_streams+self.X_cold_streams] = np.ones((1, self.X_cold_streams)) * 30
        self.X_init[:, self.X_hot_streams+self.X_cold_streams:] = np.ones((1, self.X_compressors)) * 2
        
        return self.X_init
    
    
    def _get_bounds(self):
        bound_hot_streams = (10, 40)
        bound_cold_streams = (-10, 39)
        bound_compressors = (np.sqrt(3), 3)
        
        self.bounds = [None for _ in range(self.X_dim)]
        
        # Bounds for H-1 hot stream outlet temperature
        for i in range(self.X_hot_streams):
            bound = {"name": f"x_{i}", "type": "continuous", "domain": bound_hot_streams}
            self.bounds.append(bound)
            
        # Bounds for H-1 cold streams outlet temperatures
        for j in range(self.X_hot_streams, self.X_hot_streams+self.X_cold_streams):
            bound = {"name": f"x_{j}", "type": "continuous", "domain": bound_cold_streams}
            self.bounds.append(bound)
            
        # Bounds for compressors pressure ratios
        for k in range(self.X_hot_streams+self.X_cold_streams, self.X_dim):
            bound = {"name": f"x_{k}", "type": "continuous", "domain": bound_compressors}
            self.bounds.append(bound)
            
        return self.bounds
    
    def _get_constraints(self):
        X_comp_idx = [k for k in range(self.X_hot_streams+self.X_cold_streams, self.X_dim)]
        
        X_to_prod_pressure_idx = X_comp_idx[:self.num_comp_list[0]-1]
        X_to_prod_pressure_upper = f"{'*'.join([f'x[:, {i}]' for i in X_to_prod_pressure_idx])} - {self.prod_pressure}/np.sqrt(3)/1.81325"
        X_to_prod_pressure_lower = f"{self.prod_pressure}/3/1.81325 - {'*'.join([f'x[:, {i}]' for i in X_to_prod_pressure_idx])}"
        
        X_pressurizing_idx = X_comp_idx[self.num_comp_list[0]-1:]
        X_pressurizing_upper = f"{'*'.join([f'x[:, {j}]' for j in X_pressurizing_idx])} - 100/{self.prod_pressure}"
        X_pressurizing_lower = f"80/{self.prod_pressure} - {'*'.join([f'x[:, {j}]' for j in X_pressurizing_idx])}"
        
        to_prod_pressure_upper_const = {"name": "c1", "constraint": X_to_prod_pressure_upper}
        to_prod_pressure_lower_const = {"name": "c2", "constraint": X_to_prod_pressure_lower}
        pressurizing_upper_const = {"name": "c3", "constraint": X_pressurizing_upper}
        pressurizing_lower_const = {"name": "c4", "constraint": X_pressurizing_lower}
        
        self.constraints = [to_prod_pressure_upper_const, to_prod_pressure_lower_const,
                            pressurizing_upper_const, pressurizing_lower_const]
        
        return self.constraints    
    
    def _get_X_names(self):
        self.X_names = []
        
        # Names for H-1 hot stream outlet temperature
        for i in range(1, self.X_hot_streams+1):
            name = f"T_H-1_H{i}_out"
            self.X_names.append(name)
            
        # Names for H-1 cold streams outlet temperature
        for j in range(1, self.X_cold_streams+1):
            name = f"T_H-1_C{j+1}_out"
            self.X_names.append(name)
            
        # Names for compressor pressure ratios
        for k in range(1, self.num_compressors+1):
            if k == self.num_comp_list[0]:
                pass
            else:
                name = f"PR_C-{k}"
            self.X_names.append(name)
            
        return self.X_names
    
    
    def find_node(self):
        # Nodes corresponding to X
        # H-1 hot streams
        node_block_H1_H1HOUT = "\\Data\\Blocks\\H-1\\Input\\VALUE\\H1HIN"
        
        # H-1 cold streams
        for j in range(1, self.X_cold_streams+1):
            globals()[f"node_block_H1_H1C{j+1}OUT"] = f"\\Data\\Blocks\\H-1\\Input\\VALUE\\H1C{j+1}IN"    # (C)
                
        # Compressor pressure ratios
        for k in range(1, self.num_compressors+1):
            # Excluding the compressor with the product pressure
            if k != self.num_comp_list[0]:
                globals()[f"node_block_C{k}_pres_ratio"] = f"\\Data\\Blocks\\C-{k}\\Input\\PRATIO"
            else:
                pass
            
        # Make X_idx_match dictionary
        X_idx_match = {0: [node_block_H1_H1HOUT]}
        
        for j in range(self.X_cold_streams): 
            X_idx_match[j+self.X_hot_streams] = [globals()[f"node_block_H1_H1C{j+2}OUT"]]
            
        for k in range(self.num_compressors):
            if k < self.num_comp_list[0]-1:
                X_idx_match[k+self.X_hot_streams+self.X_cold_streams] = [globals()[f"node_block_C{k+1}_pres_ratio"]]
            elif k == self.num_comp_list[0]-1:
                pass
            else:
                X_idx_match[k+self.X_hot_streams+self.X_cold_streams-1] = [globals()[f"node_block_C{k+1}_pres_ratio"]]
                
        # Nodes corresponding to Y
        # Compressor powers
        for comp_no in range(1, self.num_compressors+1):
            globals()[f"node_block_C{comp_no}_power"] = f"\\Data\\Blocks\\C-{comp_no}\\Output\\WNET"    # (kW)
            
        # Cooler duties, hot stream inlet temperatures, operating pressures
        for cooler_no in range(1, self.num_coolers+1):
            globals()[f"node_block_K{cooler_no}_duty"] = f"\\Data\\Blocks\\K-{cooler_no}\\Output\\QNET"    # (cal/sec)
            globals()[f"node_block_K{cooler_no}_hot_in"] = f"\\Data\\Streams\\K{cooler_no}IN\\Output\\TEMP_OUT\\MIXED"    # (C)
            globals()[f"node_block_K{cooler_no}_pres"] = f"\\Data\\Streams\\K{cooler_no}IN\\Output\\PRES_OUT\\MIXED"    # (bar)
            
        # H-1 LMTD and duty
        node_block_H1_LMTD = "\\Data\\Blocks\\H-1\\Output\\LMTD"    # (C)
        node_block_H1_duty = "\\Data\\Blocks\\H-1\\Output\\QCALC2"    # (cal/sec)
        
        # Vessel inlet liquid volumetric flow rates and operating pressures
        for vessel_no in range(1, self.num_vessels+1):
            if vessel_no == 1:
                globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] = "\\Data\\Streams\\WATER\\Output\\VOLFLMX_LIQ"     # (L/min)
            else:
                globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] = f"\\Data\\Streams\\D{vessel_no}IN\\Output\\VOLFLMX_LIQ"    # (L/min)
            globals()[f"node_block_D{vessel_no}_pres"] = f"\\Data\\Streams\\D{vessel_no}IN\\Output\\PRES_OUT\\MIXED"    # (bar)
            
        # Feed and product mass flow rate
        node_feed_massflow = "\\Data\\Streams\\FEED\\Output\\MASSFLMX\\MIXED"    # (kg/hr)
        node_prod_massflow = "\\Data\\Streams\\PROD\\Output\\MASSFLMX\\MIXED"    # (kg/hr)
            
        # Nodes for calculating exergy efficiency
        # Physical exergy
        node_feed_exergy_phys = "\\Data\\Streams\\FEED\\Output\\STRM_UPP\\EXERGYFL\\MIXED\\TOTAL"    # (kW)
        node_prod_exergy_phys = "\\Data\\Streams\\PROD\\Output\\STRM_UPP\\EXERGYFL\\MIXED\\TOTAL"    # (kW)
        
        # CO2 mole fraction
        node_feed_molefrac_co2 = "\\Data\\Streams\\FEED\\Output\\MOLEFRAC\\MIXED\\CO2"
        node_prod_molefrac_co2 = "\\Data\\Streams\\PROD\\Output\\MOLEFRAC\\MIXED\\CO2"
        
        # H2O mole fraction
        node_feed_molefrac_h2o = "\\Data\\Streams\\FEED\\Output\\MOLEFRAC\\MIXED\\H2O"
        node_prod_molefrac_h2o = "\\Data\\Streams\\PROD\\Output\\MOLEFRAC\\MIXED\\H2O"
        
        # CO2 mass flow rate
        node_feed_massflow_co2 = "\\Data\\Streams\\FEED\\Output\\MASSFLOW\\MIXED\\CO2"    # (kg/hr)
        node_prod_massflow_co2 = "\\Data\\Streams\\PROD\\Output\\MASSFLOW\\MIXED\\CO2"    # (kg/hr)
        
        # H2O mass flow rate
        node_feed_massflow_h2o = "\\Data\\Streams\\FEED\\Output\\MASSFLOW\\MIXED\\H2O"    # (kg/hr)
        node_prod_massflow_h2o = "\\Data\\Streams\\PROD\\Output\\MASSFLOW\\MIXED\\H2O"    # (kg/hr)
        
        # Make Y_nodes dictionary
        Y_nodes = {"Power": [globals()[f"node_block_C{comp_no}_power"] for comp_no in range(1, self.num_compressors+1)],
                   "Cooler_duty": [globals()[f"node_block_K{cooler_no}_duty"] for cooler_no in range(1, self.num_coolers+1)],
                   "Cooler_hot_in": [globals()[f"node_block_K{cooler_no}_hot_in"] for cooler_no in range(1, self.num_coolers+1)],
                   "Cooler_pres": [globals()[f"node_block_K{cooler_no}_pres"] for cooler_no in range(1, self.num_coolers+1)],
                   "Vessel_inlet_liq_vol": [globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] for vessel_no in range(1, self.num_vessels+1)],
                   "Vessel_pres": [globals()[f"node_block_D{vessel_no}_pres"] for vessel_no in range(1, self.num_vessels+1)],
                   "H1_LMTD": [node_block_H1_LMTD], 
                   "H1_duty": [node_block_H1_duty],
                   "massflow": [node_feed_massflow, node_prod_massflow],
                   "exergy_phys": [node_feed_exergy_phys, node_prod_exergy_phys],
                   "molefrac_co2": [node_feed_molefrac_co2, node_prod_molefrac_co2],
                   "molefrac_h2o": [node_feed_molefrac_h2o, node_prod_molefrac_h2o],
                   "massflow_co2": [node_feed_massflow_co2, node_prod_massflow_co2],
                   "massflow_h2o": [node_feed_massflow_h2o, node_prod_massflow_h2o]}
        
        return X_idx_match, Y_nodes


# %% OpenClosed-loop configuration
class OpenClosedLoop():
    def __init__(self, feed_flowrate, prod_pressure, comp_type, num_coolers, num_vessels, refrig):
        self.feed_flowrate = feed_flowrate
        self.prod_pressure = prod_pressure
        self.comp_type = comp_type
        self.num_coolers = num_coolers
        self.num_vessels = num_vessels
        self.refrig = refrig
        
        self.num_comp_list = list(map(int, self.comp_type.split("-")))
        self.num_compressors = np.sum(np.array(self.num_comp_list))

        # Nodes for product pressure
        if self.num_comp_list[1] != 0:
            self.prod_nodes = [f"\\Data\\Blocks\\V-{self.num_comp_list[2]}\\Input\\P_OUT"]
        else:
            self.prod_nodes = [f"\\Data\\Blocks\\V-{self.num_comp_list[2]}\\Input\\P_OUT",
                               f"\\Data\\Blocks\\C-{self.num_comp_list[0]}\\Input\\PRES"] 

        # Node for the feed flow rate
        self.feed_node = ["\\Data\\Streams\\FEED\\Input\\TOTFLOW\\MIXED"]

        # Initial value of optimization variable X
        self.X_init = self._get_X_init()

        # Bounds of optimization variable X
        self.bounds = self._get_bounds()
        
        # Variable names of X
        self.X_names = self._get_X_names()
        

    def _get_X_init(self):
        # Number of design spec for refrigerant outlet temperature spec settings
        self.X_design_spec = 1
        
        # Number of initial (C+1)th valve pressures settings
        self.X_valve_init = 1
        
        # Number of H-1 hot stream outlet temperature settings
        self.X_hot_streams = 1

        # Number of H-1 cold stream outlet temperature settings
        self.X_cold_streams = self.num_comp_list[2]

        # Number of final pressurized CO2 pressure settings
        self.X_CO2_P = 1
        
        # Number of final pressurized refrig pressure settings
        self.X_refrig_P = 1 

        # Dimension of X
        self.X_dim = self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams + self.X_CO2_P + self.X_refrig_P
        
        # Set X_init
        self.X_init = np.empty(shape=(1, self.X_dim))

        self.X_init[:, :self.X_design_spec] = 1           # (dimensionless)
        self.X_init[:, self.X_design_spec : self.X_design_spec + self.X_valve_init] = 2    # (bar)
        self.X_init[:, self.X_design_spec + self.X_valve_init : self.X_design_spec + self.X_valve_init + self.X_hot_streams] = 36 # (C)
        self.X_init[:, self.X_design_spec + self.X_valve_init + self.X_hot_streams : self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams] \
                                                                          = np.ones((1, self.X_cold_streams)) * 20 # (C) 
        self.X_init[:, self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams: self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams + self.X_CO2_P] = 50
        self.X_init[:, self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams + self.X_CO2_P:] = 20        
        return self.X_init

    def _get_bounds(self):
        bound_design_spec = (-20, 39)
        bound_valve_init = (1., 15.)
        bound_hot_streams = (10, 40)
        bound_cold_streams = (-10, 39)       
        bound_CO2_P = (30, 70)
        bound_refrig_P = (15, 40)

        self.bounds = []
        
        # Bounds for Refrigerant flow rate
        for i in range(self.X_design_spec):
            bound = {"name": f"x_{i}", "type": "continuous", "domain": bound_design_spec}
            self.bounds.append(bound)
        
        # Bounds for V-(C+1) initial pressure
        for j in range(self.X_design_spec, self.X_design_spec + self.X_valve_init):
            bound = {"name": f"x_{j}", "type": "continuous", "domain": bound_valve_init}
            self.bounds.append(bound)
            
        # Bounds for H-1 hot stream outlet temperature
        for k in range(self.X_design_spec + self.X_valve_init, self.X_design_spec + self.X_valve_init + self.X_hot_streams):
            bound = {"name": f"x_{k}", "type": "continuous", "domain": bound_hot_streams}
            self.bounds.append(bound)

        # Bounds for H-1 cold streams outlet temperatures
        for l in range(self.X_design_spec + self.X_valve_init + self.X_hot_streams, 
                       self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams):
            bound = {"name": f"x_{l}", "type": "continuous", "domain" : bound_cold_streams}
            self.bounds.append(bound)
        
        # Bounds for final pressurized CO2 pressure
        for m in range(self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams,
                       self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams + self.X_CO2_P):
            bound = {"name": f"x_{m}", "type": "continuous", "domain" : bound_CO2_P}
            self.bounds.append(bound)

        # Bounds for final pressurized refrig pressure
        for n in range(self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams + self.X_CO2_P, self.X_dim):
            bound = {"name": f"x_{n}", "type": "continuous", "domain" : bound_refrig_P}
            self.bounds.append(bound)
        
        return self.bounds

    def _get_X_names(self):
        self.X_names = []
        
        # Names for design spec (refrigerant mass flow rate change ratio)
        name = "REFRIG H-1 outlet T"
        self.X_names.append(name)
            
        # Names for V-(C+1) pressure
        name = f"p_V-{self.num_comp_list[2] + 1}"
        self.X_names.append(name)

        # Names for H-1 hot stream outlet temperature
        for i in range(1, self.X_hot_streams + 1):
            name = f"T_H-1_H{i}_out"
            self.X_names.append(name)

        # Names for H-1 cold stream outlet temperatures
        for j in range(1, self.X_cold_streams + 1):
            name = f"T_H-1_C{j}_out"
            self.X_names.append(name)

        # Names for final pressurized CO2 pressure
        name = "P_final_CO2"
        self.X_names.append(name)
    
        # Names for final pressurized refrig pressure
        name = "P_final_refrig"
        self.X_names.append(name)

        return self.X_names
    

    def find_node(self):
        # Nodes corresponding to X
        # design spec for REFRIG H-1 outlet T 
        node_design_spec = "\\Data\\Flowsheeting Options\\Design-Spec\\MULT-DS\\Input\\EXPR2"    # (C)
        
        # V-(C+1) pressure
        globals()[f"node_block_V{self.num_comp_list[2] + 1}_pres"] = f"\\Data\\Blocks\\V-{self.num_comp_list[2] + 1}\\Input\\P_OUT"    # (bar)

        # H-1 hot streams
        node_block_H1_H1HOUT = "\\Data\\Blocks\\H-1\\Input\\VALUE\\H1HIN"

        # H-1 cold streams  
        for i in range(1, self.X_cold_streams + 1):
            globals()[f"node_block_H1_H1C{i}OUT"] = f"\\Data\\Blocks\\H-1\\Input\\VALUE\\H1C{i}IN"   # (C)

        # Final pressurized CO2 pressure
        globals()[f"node_block_C{sum(self.num_comp_list[:-1])}_pres"] = f"\\Data\\Blocks\\C-{sum(self.num_comp_list[:-1])}\\Input\\PRES"

        # Final pressurized refrig pressure
        globals()[f"node_block_C{sum(self.num_comp_list)}_pres"] = f"\\Data\\Blocks\\C-{sum(self.num_comp_list)}\\Input\\PRES"
       
        # Make X_idx_match dictionary
        X_idx_match = {0: [node_design_spec],
                       1: [globals()[f"node_block_V{self.num_comp_list[2] + 1}_pres"]],
                       2: [node_block_H1_H1HOUT]}

        for i in range(self.X_cold_streams):
            X_idx_match[i + self.X_design_spec + self.X_valve_init + self.X_hot_streams] = [globals()[f"node_block_H1_H1C{i+1}OUT"]]

        X_idx_match[self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams] = [globals()[f"node_block_C{sum(self.num_comp_list[:-1])}_pres"]]

        X_idx_match[self.X_design_spec + self.X_valve_init + self.X_hot_streams + self.X_cold_streams + 1] = [globals()[f"node_block_C{sum(self.num_comp_list)}_pres"]]

        # Nodes corresponding to Y
        # Compressor powers
        for comp_no in range(1, self.num_compressors+1):
            globals()[f"node_block_C{comp_no}_power"] = f"\\Data\\Blocks\\C-{comp_no}\\Output\\WNET"    # (kW)
            
        # Cooler duties, hot stream inlet temperatures, operating pressures
        for cooler_no in range(1, self.num_coolers+1):
            globals()[f"node_block_K{cooler_no}_duty"] = f"\\Data\\Blocks\\K-{cooler_no}\\Output\\QNET"    # (cal/sec)
            globals()[f"node_block_K{cooler_no}_hot_in"] = f"\\Data\\Streams\\K{cooler_no}IN\\Output\\TEMP_OUT\\MIXED"    # (C)
            globals()[f"node_block_K{cooler_no}_pres"] = f"\\Data\\Streams\\K{cooler_no}IN\\Output\\PRES_OUT\\MIXED"    # (bar)

        # H-1 LMTD and duty

        node_block_H1_LMTD = "\\Data\\Blocks\\H-1\\Output\\LMTD"    # (C)
        node_block_H1_duty = "\\Data\\Blocks\\H-1\\Output\\QCALC2"  # (cal/sec)

        # Vessel inlet liquid volumetric flow rates and operating pressures

        for vessel_no in range(1, self.num_vessels+1):
            if vessel_no == 1:
                globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] = "\\Data\\Streams\\WATER\\Output\\VOLFLMX_LIQ" # (L/min)
            else:
                globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] = f"\\Data\\Streams\\D{vessel_no}IN\\Output\\VOLFLMX_LIQ" # (L/min)
            globals()[f"node_block_D{vessel_no}_pres"] = f"\\Data\\Streams\\D{vessel_no}IN\\Output\\PRES_OUT\\MIXED" # (bar) 

        # Feed and product mass flow rate
        node_feed_massflow = "\\Data\\Streams\\FEED\\Output\\MASSFLMX\\MIXED"    # (kg/hr)
        node_prod_massflow = "\\Data\\Streams\\PROD\\Output\\MASSFLMX\\MIXED"    # (kg/hr)
            
        # Nodes for calculating exergy efficiency
        # Physical exergy
        node_feed_exergy_phys = "\\Data\\Streams\\FEED\\Output\\STRM_UPP\\EXERGYFL\\MIXED\\TOTAL"    # (kW)
        node_prod_exergy_phys = "\\Data\\Streams\\PROD\\Output\\STRM_UPP\\EXERGYFL\\MIXED\\TOTAL"    # (kW)
        
        # CO2 mole fraction
        node_feed_molefrac_co2 = "\\Data\\Streams\\FEED\\Output\\MOLEFRAC\\MIXED\\CO2"
        node_prod_molefrac_co2 = "\\Data\\Streams\\PROD\\Output\\MOLEFRAC\\MIXED\\CO2"
        
        # H2O mole fraction
        node_feed_molefrac_h2o = "\\Data\\Streams\\FEED\\Output\\MOLEFRAC\\MIXED\\H2O"
        node_prod_molefrac_h2o = "\\Data\\Streams\\PROD\\Output\\MOLEFRAC\\MIXED\\H2O"
        
        # CO2 mass flow rate
        node_feed_massflow_co2 = "\\Data\\Streams\\FEED\\Output\\MASSFLOW\\MIXED\\CO2"    # (kg/hr)
        node_prod_massflow_co2 = "\\Data\\Streams\\PROD\\Output\\MASSFLOW\\MIXED\\CO2"    # (kg/hr)
        
        # H2O mass flow rate
        node_feed_massflow_h2o = "\\Data\\Streams\\FEED\\Output\\MASSFLOW\\MIXED\\H2O"    # (kg/hr)
        node_prod_massflow_h2o = "\\Data\\Streams\\PROD\\Output\\MASSFLOW\\MIXED\\H2O"    # (kg/hr)
        
        # Make Y_nodes dictionary
        Y_nodes = {"Power": [globals()[f"node_block_C{comp_no}_power"] for comp_no in range(1, self.num_compressors+1)],
                   "Cooler_duty": [globals()[f"node_block_K{cooler_no}_duty"] for cooler_no in range(1, self.num_coolers+1)],
                   "Cooler_hot_in": [globals()[f"node_block_K{cooler_no}_hot_in"] for cooler_no in range(1, self.num_coolers+1)],
                   "Cooler_pres": [globals()[f"node_block_K{cooler_no}_pres"] for cooler_no in range(1, self.num_coolers+1)],
                   "Vessel_inlet_liq_vol": [globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] for vessel_no in range(1, self.num_vessels+1)],
                   "Vessel_pres": [globals()[f"node_block_D{vessel_no}_pres"] for vessel_no in range(1, self.num_vessels+1)],
                   "H1_LMTD" : [node_block_H1_LMTD],
                   "H1_duty" : [node_block_H1_duty], 
                   "massflow": [node_feed_massflow, node_prod_massflow],
                   "exergy_phys": [node_feed_exergy_phys, node_prod_exergy_phys],
                   "molefrac_co2": [node_feed_molefrac_co2, node_prod_molefrac_co2],
                   "molefrac_h2o": [node_feed_molefrac_h2o, node_prod_molefrac_h2o],
                   "massflow_co2": [node_feed_massflow_co2, node_prod_massflow_co2],
                   "massflow_h2o": [node_feed_massflow_h2o, node_prod_massflow_h2o]}
        
        return X_idx_match, Y_nodes


# %% Closed-loop configuration
class ClosedLoop():
    def __init__(self, feed_flowrate, prod_pressure, comp_type, num_coolers, num_vessels, refrig):
        self.feed_flowrate = feed_flowrate
        self.prod_pressure = prod_pressure
        self.comp_type = comp_type
        self.num_coolers = num_coolers
        self.num_vessels = num_vessels
        self.num_comp_list = list(map(int, self.comp_type.split("-")))
        self.num_compressors = np.sum(np.array(self.num_comp_list))
        self.refrig = refrig
        
        # Nodes for product pressure
        self.prod_nodes = [f"\\Data\\Blocks\\C-{self.num_comp_list[0]}\\Input\\PRES"]
        
        # Node for the feed flow rate
        self.feed_node = ["\\Data\\Streams\\FEED\\Input\\TOTFLOW\\MIXED"]
        
        # Initial value of optimization variable X
        self.X_init = self._get_X_init()
        
        # Bounds of optimization variable X
        self.bounds = self._get_bounds()
        
        # Constraints of optimization variables X
        self.constraints = self._get_constraints()
        
        # Variable names of X
        self.X_names = self._get_X_names()
        
        
    def _get_X_init(self):
        # Number of refrigerant flow rates settings
        self.X_refrig = 1
        
        # Number of initial valve pressures settings
        self.X_valve_init = 1
        
        # Number of compressors pressure ratios settings
        self.X_compressors = self.num_compressors - 1
        
        # Dimension of X
        self.X_dim = self.X_refrig + self.X_valve_init + self.X_compressors
        
        # Set X_init
        self.X_init = np.empty(shape=(1, self.X_dim))
        self.X_init[:, :self.X_refrig] = 3.456 * self.feed_flowrate    # (tonne/day)
        self.X_init[:, self.X_refrig:self.X_refrig+self.X_valve_init] = 4    # (bar)
        self.X_init[:, self.X_refrig+self.X_valve_init:] = np.ones((1, self.X_compressors)) * 2
        
        return self.X_init
    
    
    def _get_bounds(self):
        bound_refrig = (self.feed_flowrate*2.5, self.feed_flowrate*4.0)
        bound_valve_init = (1., 15.)
        bound_compressors = (np.sqrt(3), 3)
        
        self.bounds = []
        
        # Bounds for Refrigerant flow rate
        for i in range(self.X_refrig):
            bound = {"name": f"x_{i}", "type": "continuous", "domain": bound_refrig}
            self.bounds.append(bound)
        
        # Bounds for V-1 initial pressure
        for j in range(self.X_refrig, self.X_refrig+self.X_valve_init):
            bound = {"name": f"x_{j}", "type": "continuous", "domain": bound_valve_init}
            self.bounds.append(bound)
            
        # Bounds for compressors pressure ratios
        for k in range(self.X_refrig+self.X_valve_init, self.X_dim):
            bound = {"name": f"x_{k}", "type": "continuous", "domain": bound_compressors}
            self.bounds.append(bound)
            
        return self.bounds
    
    def _get_constraints(self):
        X_comp_idx = [k for k in range(self.X_refrig+self.X_valve_init, self.X_dim)]
        
        X_to_prod_pressure_idx = X_comp_idx[:self.num_comp_list[0]-1]
        X_to_prod_pressure_upper = f"{'*'.join([f'x[:, {i}]' for i in X_to_prod_pressure_idx])} - {self.prod_pressure}/np.sqrt(3)/1.81325"
        X_to_prod_pressure_lower = f"{self.prod_pressure}/3/1.81325 - {'*'.join([f'x[:, {i}]' for i in X_to_prod_pressure_idx])}"
        
        to_prod_pressure_upper_const = {"name": "c1", "constraint": X_to_prod_pressure_upper}
        to_prod_pressure_lower_const = {"name": "c2", "constraint": X_to_prod_pressure_lower}
        
        X_refrig_loop_idx = X_comp_idx[self.num_comp_list[0]-1:]
        X_refrig_loop_upper = f"{'*'.join([f'x[:, {k}]' for k in X_refrig_loop_idx])} - 100/x[:, 1]"
                    
        if self.refrig == "CO2":
            X_refrig_loop_lower = f"80/x[:,1] - {'*'.join([f'x[:, {k}]' for k in X_refrig_loop_idx])}"
        elif self.refrig == "NH3":
            X_refrig_loop_lower = f"15/x[:,1] - {'*'.join([f'x[:, {k}]' for k in X_refrig_loop_idx])}"
        elif self.refrig == "C3H6":
            X_refrig_loop_lower = f"15/x[:,1] - {'*'.join([f'x[:, {k}]' for k in X_refrig_loop_idx])}"
        
        to_prod_pressure_upper_const = {"name": "c1", "constraint": X_to_prod_pressure_upper}
        to_prod_pressure_lower_const = {"name": "c2", "constraint": X_to_prod_pressure_lower}
        refrig_loop_upper_const = {"name" : "c5", "constraint" : X_refrig_loop_upper}
        refrig_loop_lower_const = {"name" : "c6", "constraint" : X_refrig_loop_lower}
        
        self.constraints = [to_prod_pressure_upper_const, to_prod_pressure_lower_const,
                            refrig_loop_upper_const, refrig_loop_lower_const]
        
        return self.constraints
    
    def _get_X_names(self):
        self.X_names = []
        
        # Names for refrigerant flow rate
        name = "massflow_refrig"
        self.X_names.append(name)
            
        # Names for V-1 pressure
        name = "p_V-1"
        self.X_names.append(name)
            
        # Names for compressor pressure ratios
        for k in range(1, self.num_compressors+1):
            if k == self.num_comp_list[0]:
                pass
            else:
                name = f"PR_C-{k}"
            self.X_names.append(name)
    
        return self.X_names
    
    
    def find_node(self):
        # Nodes corresponding to X
        # Refrigerant flow rate
        node_refrig_massflow = "\\Data\\Streams\\REFRIG\\Input\\TOTFLOW\\MIXED"    # (tonne/day)
        
        # V-1 initial pressure
        node_block_V1_pres = "\\Data\\Blocks\\V-1\\Input\\P_OUT"    # (bar)
                
        # Compressor pressure ratios
        for k in range(1, self.num_compressors+1):
            # Excluding the compressor with the product pressure
            if k != self.num_comp_list[0]:
                globals()[f"node_block_C{k}_pres_ratio"] = f"\\Data\\Blocks\\C-{k}\\Input\\PRATIO"
            else:
                pass
            
        # Make X_idx_match dictionary
        X_idx_match = {0: [node_refrig_massflow],
                       1: [node_block_V1_pres]}
            
        for k in range(self.num_compressors):
            if k < self.num_comp_list[0]-1:
                X_idx_match[k+self.X_refrig+self.X_valve_init] = [globals()[f"node_block_C{k+1}_pres_ratio"]]
            elif k == self.num_comp_list[0]-1:
                pass
            else:
                X_idx_match[k+self.X_refrig+self.X_valve_init-1] = [globals()[f"node_block_C{k+1}_pres_ratio"]]
                
        # Nodes corresponding to Y
        # Compressor powers
        for comp_no in range(1, self.num_compressors+1):
            globals()[f"node_block_C{comp_no}_power"] = f"\\Data\\Blocks\\C-{comp_no}\\Output\\WNET"    # (kW)
            
        # Cooler duties, hot stream inlet temperatures, operating pressures
        for cooler_no in range(1, self.num_coolers+1):

            globals()[f"node_block_K{cooler_no}_duty"] = f"\\Data\\Blocks\\K-{cooler_no}\\Output\\QNET"    # (Gcal/hr)
            globals()[f"node_block_K{cooler_no}_hot_in"] = f"\\Data\\Streams\\K{cooler_no}IN\\Output\\TEMP_OUT\\MIXED"    # (C)
            globals()[f"node_block_K{cooler_no}_pres"] = f"\\Data\\Streams\\K{cooler_no}IN\\Output\\PRES_OUT\\MIXED"    # (bar)
            
        # H-1 LMTD and duty
        node_block_H1_LMTD = "\\Data\\Blocks\\H-1\\Output\\LMTD"    # (C)
        node_block_H1_duty = "\\Data\\Blocks\\H-1\\Output\\QCALC2"    # (Gcal/hr)
        
        # Vessel inlet liquid volumetric flow rates and operating pressures
        for vessel_no in range(1, self.num_vessels+1):
            if vessel_no == 1:
                globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] = "\\Data\\Streams\\WATER\\Output\\VOLFLMX_LIQ" # (cum/hr)
            
            else:
                globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] = f"\\Data\\Streams\\D{vessel_no}IN\\Output\\VOLFLMX_LIQ" # (cum/hr)            
            globals()[f"node_block_D{vessel_no}_pres"] = f"\\Data\\Streams\\D{vessel_no}IN\\Output\\PRES_OUT\\MIXED"    # (bar)
            
        # Feed and product mass flow rate
        node_feed_massflow = "\\Data\\Streams\\FEED\\Output\\MASSFLMX\\MIXED"    # (kg/hr)
        node_prod_massflow = "\\Data\\Streams\\PROD\\Output\\MASSFLMX\\MIXED"    # (kg/hr)
            
        # Nodes for calculating exergy efficiency
        # Physical exergy
        node_feed_exergy_phys = "\\Data\\Streams\\FEED\\Output\\STRM_UPP\\EXERGYFL\\MIXED\\TOTAL"    # (kW)
        node_prod_exergy_phys = "\\Data\\Streams\\PROD\\Output\\STRM_UPP\\EXERGYFL\\MIXED\\TOTAL"    # (kW)
        
        # CO2 mole fraction
        node_feed_molefrac_co2 = "\\Data\\Streams\\FEED\\Output\\MOLEFRAC\\MIXED\\CO2"
        node_prod_molefrac_co2 = "\\Data\\Streams\\PROD\\Output\\MOLEFRAC\\MIXED\\CO2"
        
        # H2O mole fraction
        node_feed_molefrac_h2o = "\\Data\\Streams\\FEED\\Output\\MOLEFRAC\\MIXED\\H2O"
        node_prod_molefrac_h2o = "\\Data\\Streams\\PROD\\Output\\MOLEFRAC\\MIXED\\H2O"
        
        # CO2 mass flow rate
        node_feed_massflow_co2 = "\\Data\\Streams\\FEED\\Output\\MASSFLOW\\MIXED\\CO2"    # (kg/hr)
        node_prod_massflow_co2 = "\\Data\\Streams\\PROD\\Output\\MASSFLOW\\MIXED\\CO2"    # (kg/hr)
        
        # H2O mass flow rate
        node_feed_massflow_h2o = "\\Data\\Streams\\FEED\\Output\\MASSFLOW\\MIXED\\H2O"    # (kg/hr)
        node_prod_massflow_h2o = "\\Data\\Streams\\PROD\\Output\\MASSFLOW\\MIXED\\H2O"    # (kg/hr)
        
        # Make Y_nodes dictionary
        Y_nodes = {"Power": [globals()[f"node_block_C{comp_no}_power"] for comp_no in range(1, self.num_compressors+1)],
                   "Cooler_duty": [globals()[f"node_block_K{cooler_no}_duty"] for cooler_no in range(1, self.num_coolers+1)],
                   "Cooler_hot_in": [globals()[f"node_block_K{cooler_no}_hot_in"] for cooler_no in range(1, self.num_coolers+1)],
                   "Cooler_pres": [globals()[f"node_block_K{cooler_no}_pres"] for cooler_no in range(1, self.num_coolers+1)],
                   "Vessel_inlet_liq_vol": [globals()[f"node_block_D{vessel_no}_inlet_liq_vol"] for vessel_no in range(1, self.num_vessels+1)],
                   "Vessel_pres": [globals()[f"node_block_D{vessel_no}_pres"] for vessel_no in range(1, self.num_vessels+1)],
                   "H1_LMTD": [node_block_H1_LMTD], 
                   "H1_duty": [node_block_H1_duty],
                   "massflow": [node_feed_massflow, node_prod_massflow],
                   "exergy_phys": [node_feed_exergy_phys, node_prod_exergy_phys],
                   "molefrac_co2": [node_feed_molefrac_co2, node_prod_molefrac_co2],
                   "molefrac_h2o": [node_feed_molefrac_h2o, node_prod_molefrac_h2o],
                   "massflow_co2": [node_feed_massflow_co2, node_prod_massflow_co2],
                   "massflow_h2o": [node_feed_massflow_h2o, node_prod_massflow_h2o]}
        
        return X_idx_match, Y_nodes