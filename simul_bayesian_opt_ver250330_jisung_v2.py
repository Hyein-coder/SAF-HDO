# %% Load packages
import os
import time
import random
import win32com.client as win32

import numpy as np
import matplotlib.pyplot as plt
from pickle import dump, load

import GPy
import GPyOpt
from GPyOpt.methods import ModularBayesianOptimization

from configuration_ver250330_jisung_v2 import OpenLoop, OpenClosedLoop, ClosedLoop

seed = 42
random.seed(seed)
np.random.seed(seed)

folder_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder_dir)


# %% Bayesian optimization
class SimulBayesianOpt:
    def __init__(self, feed_flowrate, simul_case, prod_pressure, comp_type = None, num_coolers = None, num_vessels = None, refrig = None):
        self.feed_flowrate = feed_flowrate    # 10, 200, 1000, 2000, 3000 (TPD) 
        self.simul_case = simul_case          # Closed_loop / Open&Closed_loop / Open_loop
        self.prod_pressure = prod_pressure    # 7.5bar, 20bar
        self.comp_type = comp_type            # "2-3", "2-4", ...
        self.refrig = refrig                  # None / CO2 / NH3 / C3H6 / C2H6 
        
        self.num_comp_list = list(map(int, comp_type.split("-")))    # Number of compressors for each zone -> list of int
        self.num_coolers = num_coolers    # Number of coolers in the simulation -> int
        self.num_vessels = num_vessels    # Number of vessels in the simulation -> int
        self.num_compressors = np.sum(np.array(self.num_comp_list))    # Number of compressors (total) -> int
        
        self.print_log = open("Print_log.txt", "w")
                
        # Define simulation classes
        if self.simul_case == "Open_loop":
            self.simul = OpenLoop(self.feed_flowrate, self.prod_pressure, self.comp_type, self.num_coolers, self.num_vessels, self.refrig)
        elif self.simul_case == "Open&Closed_loop":
             self.simul = OpenClosedLoop(self.feed_flowrate, self.prod_pressure, self.comp_type, self.num_coolers, self.num_vessels, self.refrig)
        elif self.simul_case == "Closed_loop":
            self.simul = ClosedLoop(self.feed_flowrate, self.prod_pressure, self.comp_type, self.num_coolers, self.num_vessels, self.refrig)
        else:
            raise ValueError("Simulation case should be one of 'Open_loop', 'Open&Closed_loop', or 'Closed_loop'.")
        
        if self.refrig is None:
            self.simul_file = f"{self.simul_case}/{self.simul_case}_carbon_clean_comp{self.comp_type}_cooler{self.num_coolers}_vessel{self.num_vessels}.bkp"
        else:
            self.simul_file = f"{self.simul_case}_{self.refrig}/{self.simul_case}_carbon_clean_comp{self.comp_type}_cooler{self.num_coolers}_vessel{self.num_vessels}_refrig_{self.refrig}.bkp"
        
        # Link Aspen Plus simulation file
        self.aspen = win32.Dispatch("Apwn.Document")
        self.aspen.InitFromArchive2(os.path.abspath(self.simul_file))
        self.aspen.Visible = False
        self.aspen.SuppressDialogs = True
        
        # Set the number of iterations
        self.n_iter = 100
        
        # Reference properties of substances 
        self.mw_co2 = 0.0440095                          # (kg/mol)
        self.mw_h2o = 0.01801534                         # (kg/mol)
        self.exergy_co2_gas_ref = 19.48 / self.mw_co2    # (kJ/kg)
        self.exergy_h2o_gas_ref = 9.5 / self.mw_h2o      # (kJ/kg)
        self.exergy_h2o_liq_ref = 0.9 / self.mw_h2o      # (kJ/kg)
        self.temperature_ref = 298.15                    # (K)
        self.pressure_ref = 1.01325                      # (bar)
        self.gas_const = 0.008314                        # (kJ/mol-K)

        # Reference properties for cost calculation
        self.mechanical_efficiency = 1.0
        self.CEPCI_2023 = 798       # Target year CEPCI
        self.CEPCI_2001 = 397       # Reference year CEPCI
        self.plant_year = 20        # (yrs)
        self.interest_rate = 0.07
        self.sizing_margin = 1.1    # Margin for the sizing of equipment
        
        # Overall heat transfer coefficients for each heat exchanger (cal/sec-sqcm-K)
        self.ST_U =  0.0203019
        
        self.plate_MSHE_U = {"Open_loop_carbon_clean_comp2-3_cooler5_vessel4" : 50.83776515/10000,
                             "Open_loop_carbon_clean_comp2-4_cooler6_vessel5" : 44.066142168/10000,
                             "Open_loop_carbon_clean_comp3-2_cooler5_vessel3" : 44.135958402/10000,
                             "Open_loop_carbon_clean_comp4-2_cooler6_vessel3" : 44.135958402/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-3-2_cooler7_vessel6_refrig_NH3" : 46.26786226/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-3-3_cooler8_vessel7_refrig_NH3" : 42.58041801/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-4-2_cooler8_vessel7_refrig_NH3" : 23.59022469/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-4-3_cooler9_vessel8_refrig_NH3" : 32.32775627/10000,
                             "Open&Closed_loop_carbon_clean_comp3-1-2-3_cooler9_vessel5_refrig_NH3" : 36.846749893/10000,
                             "Open&Closed_loop_carbon_clean_comp3-1-3-3_cooler10_vessel6_refrig_NH3" : 39.978497719/10000,
                             "Open&Closed_loop_carbon_clean_comp3-2-2-3_cooler10_vessel5_refrig_NH3" : 34.884071744/10000,
                             "Open&Closed_loop_carbon_clean_comp4-1-2-3_cooler10_vessel5_refrig_NH3" : 37.35582882/10000,
                             "Open&Closed_loop_carbon_clean_comp4-1-3-3_cooler11_vessel6_refrig_NH3" : 35.237046969/10000,
                             "Open&Closed_loop_carbon_clean_comp4-2-2-3_cooler11_vessel5_refrig_NH3" : 34.884071744/10000,
                             "Open&Closed_loop_carbon_clean_comp3-0-2-3_cooler8_vessel5_refrig_NH3" : 32.609956366/10000,
                             "Open&Closed_loop_carbon_clean_comp4-0-2-3_cooler9_vessel5_refrig_NH3" : 32.609956366/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-3-2_cooler7_vessel6_refrig_CO2" : 35.05733328/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-3-3_cooler8_vessel7_refrig_CO2" : 31.43718062/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-4-2_cooler8_vessel7_refrig_CO2" : 23.09039191/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-4-3_cooler9_vessel8_refrig_CO2" : 23.85426644/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-3-2_cooler7_vessel6_refrig_C3H6" : 36.26881544/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-3-3_cooler8_vessel7_refrig_C3H6" : 43.39686373/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-4-2_cooler8_vessel7_refrig_C3H6" : 23.05847881/10000,
                             "Open&Closed_loop_carbon_clean_comp2-0-4-3_cooler9_vessel8_refrig_C3H6" : 20.472508543/10000,
                             "Closed_loop_carbon_clean_comp2-3_cooler5_vessel4_refrig_C2H6" : 154.0515413/10000,
                             "Closed_loop_carbon_clean_comp2-4_cooler6_vessel5_refrig_C2H6" : 148.2420279/10000,
                             "Closed_loop_carbon_clean_comp3-3_cooler6_vessel4_refrig_C2H6" : 152.3075283/10000,
                             "Closed_loop_carbon_clean_comp4-3_cooler7_vessel4_refrig_C2H6" : 152.3075283/10000,
                             "Closed_loop_carbon_clean_comp3-4_cooler7_vessel5_refrig_C2H6" : 85.08401635/10000,
                             "Closed_loop_carbon_clean_comp4-4_cooler8_vessel5_refrig_C2H6" : 125.1097554/10000,
                             "Closed_loop_carbon_clean_comp3-3_cooler6_vessel4_refrig_NH3" : 406.03802427/10000,
                             "Closed_loop_carbon_clean_comp4-3_cooler7_vessel4_refrig_NH3" : 406.03802427/10000,
                             "Closed_loop_carbon_clean_comp3-4_cooler7_vessel5_refrig_NH3" : 374.66023/10000,
                             "Closed_loop_carbon_clean_comp4-4_cooler8_vessel5_refrig_NH3" : 374.66023/10000,
                             "Closed_loop_carbon_clean_comp2-3_cooler5_vessel4_refrig_CO2" : 176.2263396/10000,
                             "Closed_loop_carbon_clean_comp2-4_cooler6_vessel5_refrig_CO2" : 112.9519449/10000,
                             "Closed_loop_carbon_clean_comp3-3_cooler6_vessel4_refrig_CO2" : 274.3921528/10000,
                             "Closed_loop_carbon_clean_comp4-3_cooler7_vessel4_refrig_CO2" : 274.3921528/10000,
                             "Closed_loop_carbon_clean_comp3-4_cooler7_vessel5_refrig_CO2" : 262.2060741/10000,
                             "Closed_loop_carbon_clean_comp4-4_cooler8_vessel5_refrig_CO2" : 223.4102775/10000,
                             "Closed_loop_carbon_clean_comp3-3_cooler6_vessel4_refrig_C3H6" : 200.430123/10000,
                             "Closed_loop_carbon_clean_comp4-3_cooler7_vessel4_refrig_C3H6" : 75.5392657/10000,
                             "Closed_loop_carbon_clean_comp3-4_cooler7_vessel5_refrig_C3H6" : 140.6856687/10000,
                             "Closed_loop_carbon_clean_comp4-4_cooler8_vessel5_refrig_C3H6" : 140.6856687/10000
                             }
        
        self.plate_cooler_U = {"Open_loop_carbon_clean_comp2-3_cooler5_vessel4" : 353.08588898/10000,
                               "Open_loop_carbon_clean_comp2-4_cooler6_vessel5" : 322.96742142/10000,
                               "Open_loop_carbon_clean_comp3-2_cooler5_vessel3" : 152.76583548/10000,
                               "Open_loop_carbon_clean_comp4-2_cooler6_vessel3" : 152.76583548/10000,
                               "Closed_loop_carbon_clean_comp2-3_cooler5_vessel4" : 95.801089137/10000,
                               "Closed_loop_carbon_clean_comp2-4_cooler6_vessel5" : 95.801089137/10000,
                               "Closed_loop_carbon_clean_comp3-3_cooler6_vessel4" : 176.41157925/10000,
                               "Closed_loop_carbon_clean_comp3-4_cooler7_vessel5" : 176.41157925/10000,
                               "Closed_loop_carbon_clean_comp4-3_cooler7_vessel4" : 76.884494124/10000,
                               "Closed_loop_carbon_clean_comp4-4_cooler8_vessel5" : 76.884494124/10000,
                               "Open&Closed_loop_carbon_clean_comp2-0-3-2_cooler7_vessel6" : 362.16203306/10000,
                               "Open&Closed_loop_carbon_clean_comp2-0-3-3_cooler8_vessel7" : 362.16203306/10000,
                               "Open&Closed_loop_carbon_clean_comp2-0-4-2_cooler8_vessel7" : 351.6528136/10000,
                               "Open&Closed_loop_carbon_clean_comp2-0-4-3_cooler9_vessel8" : 351.6528136/10000,
                               "Open&Closed_loop_carbon_clean_comp3-1-2-3_cooler9_vessel5" : 317.4739658/10000,
                               "Open&Closed_loop_carbon_clean_comp3-1-3-3_cooler10_vessel6" : 191.98433171/10000,
                               "Open&Closed_loop_carbon_clean_comp3-2-2-3_cooler10_vessel5" : 310.5474348/10000,
                               "Open&Closed_loop_carbon_clean_comp4-1-2-3_cooler10_vessel5" : 319.40861756/10000,
                               "Open&Closed_loop_carbon_clean_comp4-1-3-3_cooler11_vessel6" : 199.6274004/10000,
                               "Open&Closed_loop_carbon_clean_comp4-2-2-3_cooler11_vessel5" : 310.5474348/10000,
                               "Open&Closed_loop_carbon_clean_comp3-0-2-3_cooler8_vessel5" : 329.36849145/10000,
                               "Open&Closed_loop_carbon_clean_comp4-0-2-3_cooler9_vessel5" : 329.36849145/10000,}
        
        # X_sample, Y_sample history
        self.history = {"iter" : 0,
                        "X_sample" : np.empty(shape=(0, self.simul.X_init.shape[1])),
                        "Y_sample" : np.empty(shape=(0, 1)),
                        "Y_sample_turton" : np.empty(shape=(0, 1)),
                        "X_sample_warning" : np.empty(shape=(0, self.simul.X_init.shape[1])),
                        "Y_sample_warning" : np.empty(shape=(0, 1)),
                        "Y_sample_warning_turton" : np.empty(shape=(0, 1)),
                        "X_sample_error" : np.empty(shape=(0, self.simul.X_init.shape[1])),
                        "Y_sample_error" : np.empty(shape=(0, 1)),
                        "Y_sample_error_turton" : np.empty(shape=(0, 1)),
                        "Exergy_efficiency" : np.empty(shape=(0, 1)),
                        "REFRIG_TPD" : np.empty(shape=(0, 1))}
        
        self.history_cost = {"iter" : 0}
        self.history_cost_breakdown = {"iter" : 0}
        self.history_extrapolation = {"iter" : 0}
        
    def run_bayesian_opt(self, temp_save_filename, *args):
        """
        Runs Bayesian optimization by GPyOpt package.

        Parameters
        ----------
        temp_save_filename : string
            File name for saving temporal results of Bayesian optimization.
        *args : dict
            Temporally saved Bayesian optimization results (optional).

        Returns
        -------
        history : dict
            Bayesian optimization result history.

        """     
        
        self.temp_save_filename = temp_save_filename
        
        # Make path for temporally saved figure
        figure_save_path = "temp_saved_fig/" + self.temp_save_filename
        if not os.path.exists("temp_saved_fig"):
            os.mkdir("temp_saved_fig")
        if not os.path.exists(figure_save_path):
            os.mkdir(figure_save_path)
            
        # Make path for temporally saved results
        result_save_path = "temp_saved_results/"
        if not os.path.exists(result_save_path):
            os.mkdir(result_save_path)
            
        # Print current simulation option
        print("\n"+"="*20, file = self.print_log)
        print("Running Bayesian optimization for ...", file = self.print_log)
        print(f"Configuration            | {self.simul_case}", file = self.print_log)
        print(f"Feed flowrate            | {self.feed_flowrate}TPD", file = self.print_log)
        print(f"Product pressure         | {self.prod_pressure}bar", file = self.print_log)
        print(f"Compressor configuration | {self.comp_type}", file = self.print_log)
        print("="*20, file = self.print_log)
        
        # If starting Bayesian optimization from temporally saved result
        if args:
            saved_data_name = args[0]
            
            with open(f"temp_saved_results/{saved_data_name}.pkl", "rb") as f:
                saved_data_dict = load(f)
                
            self.history = saved_data_dict
            start_iter = self.history["iter"] + 1
            X_sample = self.history["X_sample"]
            
            print("\nLoaded temporally saved data ...", file = self.print_log)
            print(f"Starting from iteration {start_iter+1}", file = self.print_log)
            
        # If starting Bayesian optimization from the beginning
        else:
            start_iter = 0
            self.history["iter"] = start_iter
            
            X_sample = self.simul.X_init
            
        # Settings for GPyOpt Bayesian optimization
        # Domains for variables
        domain = GPyOpt.Design_space(space=self.simul.bounds)
        
        # Kernel function for Bayesian optimization: m52
        m52 = GPy.kern.Matern52(input_dim=self.simul.X_init.shape[1], variance=1.0, lengthscale=1.0)
        
        # Objective function: Levelized cost
        objective = GPyOpt.core.task.SingleObjective(self.obj_levelized_cost)
        
        # Gaussian process regression model
        model = GPyOpt.models.GPModel(kernel=m52, exact_feval=True,
                                      optimize_restarts=25, verbose=False)
        
        # Acquisition function: Expected improvement (EI)
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(domain)
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, domain, optimizer=acquisition_optimizer)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        
        # Initial point of GPyOpt
        initial_design = X_sample
   
        # Set Bayesian optimization
        BO = ModularBayesianOptimization(model, domain, objective, acquisition, evaluator, initial_design)

        # Maximum time and iterations for Bayesian optimization
        max_time = None
        max_iter = self.n_iter
        tolerance = -np.inf
        
        # Run Bayesian optimization
        BO.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False)
        
        time.sleep(1)
        self.aspen.Close()
        self.print_log.close()
                
        return self.history
    
    
    def check_convergence(self):
        """
        Check the simulation results, 
        and categorize as error for infeasible results.

        Returns
        -------
        status : string
            Simulation results (Converged / Warning / Error).

        """
        # Infeasible product stream: If vapor flow rate of PROD is not zero
        if self.aspen.Tree.FindNode("\\Data\\Streams\\PROD\\Output\\VFRAC_OUT\\MIXED").Value != 0:
            print(f"Iteration {self.history['iter']} : Aspen Plus Simulation had error in liquefying PROD stream.", file = self.print_log)
            status = "Error"
            
        # Infeasible temperature at the compressor outlet:
        # If temperature at the compressor outlet is larger than 150C
        elif any(self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{comp_no}\\Output\\TOC").Value == None  or \
             self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{comp_no}\\Output\\TOC").Value > 150 
                 for comp_no in range(1, self.num_compressors+1)):
            print(f"Iteration {self.history['iter']} : Aspen Plus Simulation had error in compressor outlet temperature (compressor outlet T over 150 C or None)", file = self.print_log)
            status = "Error"
            
        # Infeasible pressure at the compressor outlet:
        # If pressure at the compressor outlet is larger than 100bar
        elif any(self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{comp_no}\\Output\\POC").Value == None or \
             self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{comp_no}\\Output\\POC").Value > 100
                 for comp_no in range(1, self.num_compressors+1)):
            print(f"Iteration {self.history['iter']} : Aspen Plus Simulation had error in compressor outlet pressure (compressor outlet P over 100 bar or None)", file = self.print_log)
            status = "Error"
            
        # Convergence check for feasible values
        else:
            # Check convergence from the result summary
            is_err_or_warn = "\\Data\\Results Summary\\Run-Status\\Output\\PER_ERROR"
            if self.aspen.Tree.FindNode(is_err_or_warn).Value != 0:
                errmsg = "Results Summary status: \n"
                for num_line in range(1,100):
                    if self.aspen.Tree.FindNode(is_err_or_warn + "\\" + str(num_line)) is None:
                        break
                    else:
                        errmsg = errmsg + "\n" + self.aspen.Tree.FindNode(is_err_or_warn + "\\" + str(num_line)).Value
                
                not_err_but_warn = "warnings" in errmsg and "errors" not in errmsg
                status = 'Warning' if not_err_but_warn else 'Error'

                # Simulation converged with warnings
                if not_err_but_warn is True:
                    
                    # Consider temperature crossover as an error
                    warnmsgH_1 = "H-1 Warning messages: \n"
                    if "H-1" in errmsg:
                        for num_line in range(1,100):
                            if self.aspen.Tree.FindNode("\\Data\\Blocks\\H-1\\Output\\PER_ERROR\\" + str(num_line)) is None:
                                break
                            else:
                                warnmsgH_1 = warnmsgH_1 + "\n" + self.aspen.Tree.FindNode("\\Data\\Blocks\\H-1\\Output\\PER_ERROR\\" + str(num_line)).Value
                        if "CROSSOVER" in warnmsgH_1:
                            print("-"*30, file = self.print_log)
                            print(f"Iteration {self.history['iter']}: Aspen Plus Simulation had error because of temperature crossover.", file = self.print_log)
                            print("-"*30 + "\n" + warnmsgH_1 + "\n" + "-"*30, file = self.print_log)
                            status = "Error"       
                            
                        else:
                            print("-"*30, file = self.print_log)
                            print(f"Iteration {self.history['iter']}: Aspen Plus Simulation converged with warnings.", file = self.print_log)
                            print("-"*30 + "\n" + errmsg + "\n" + "-"*30, file = self.print_log)     
                            status = "Warning"
                    else:
                        print("-"*30, file = self.print_log)
                        print(f"Iteration {self.history['iter']}: Aspen Plus Simulation converged with warnings.", file = self.print_log)
                        print("-"*30 + "\n" + errmsg + "\n" + "-"*30, file = self.print_log)    
                        status = "Warning"
                        
                else:
                    print("-"*30, file = self.print_log)
                    print(f"Iteration {self.history['iter']}: Aspen Plus Simulation did not converge.", file = self.print_log)
                    print("-"*30 + "\n" + errmsg + "\n" + "-"*30, file = self.print_log)    
                    status = "Error"
            
            else:   # Simulation converged with no warnings
                print(f"Iteration {self.history['iter']} : Aspen Plus Simulation converged", file = self.print_log)
                status = "Converged"    
            
        return status
    
    
    def obj_levelized_cost(self, X):
        """
        Calculates levelized cost of the product liquefied CO2 from the plant.

        Parameters
        ----------
        X : numpy.ndarray
            Decision point to calculate the levelized cost.

        Returns
        -------
        levelized_cost : float
            Levelized cost of the product liquefied CO2 from the plant.

        """
        X = np.atleast_2d(X)
        feed_pressure = 1.81325 # Unit : [bar]
        
        # Set feed flow rate
        for node in self.simul.feed_node:
            self.aspen.Tree.FindNode(node).Value = self.feed_flowrate
            
        # Set product pressure
        for node in self.simul.prod_nodes:
            self.aspen.Tree.FindNode(node).Value = self.prod_pressure
            
        # Optimization variables
        X_idx_match, Y_nodes = self.simul.find_node()
        
        for i in X_idx_match.keys():
            param_to_change = len(X_idx_match[i])
            
            for j in range(param_to_change):
                self.aspen.Tree.FindNode(X_idx_match[i][j]).Value = X[0][i].item()
              
        # Match pressure conditions
        # Open-loop
        if self.simul_case == "Open_loop":
            for valve_no in range(self.num_comp_list[-1], 0, -1):
                if valve_no == self.num_comp_list[-1]:
                    globals()[f"V{valve_no}_pres"] = self.prod_pressure
                else:
                    globals()[f"V{valve_no}_pres"] = globals()[f"V{valve_no+1}_pres"] * X[0][-1 - valve_no]
                    
                self.aspen.Tree.FindNode(f"\\Data\\Blocks\\V-{valve_no}\\Input\\P_OUT").Value = \
                    globals()[f"V{valve_no}_pres"]
        
        # Open&closed-loop
        elif self.simul_case == "Open&Closed_loop":
            if self.num_comp_list[1] == 0:
                globals()[f"C{self.num_comp_list[0]}_pres"] = self.prod_pressure
            else:
                globals()[f"C{self.num_comp_list[0]}_pres"] = self.prod_pressure * \
                                                                np.prod(X[0][2 + self.num_comp_list[0] + self.num_comp_list[2] : 
                                                                        2 + self.num_comp_list[0] + self.num_comp_list[1] + self.num_comp_list[2]])
            self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{self.num_comp_list[0]}\\Input\\PRES").Value = globals()[f"C{self.num_comp_list[0]}_pres"]

            PR1 = (globals()[f"C{self.num_comp_list[0]}_pres"] / feed_pressure)**(1/(self.num_comp_list[0]))
            PR23 = (X[0][-2] / self.prod_pressure)**(1/(self.num_comp_list[1] + self.num_comp_list[2]))
            PR4 = (X[0][-1] / X[0][1])**(1/self.num_comp_list[3])
            
            # Initial compressing stage (A) PR settings
            for comp_no in range(1, self.num_comp_list[0]):
                self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{comp_no}\\Input\\PRATIO").Value = PR1

            # Second and Third compressing stage (B, C) PR settings
            for comp_no in range(self.num_comp_list[0]+1, self.num_comp_list[0] + self.num_comp_list[1] + self.num_comp_list[2]):
                self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{comp_no}\\Input\\PRATIO").Value = PR23
                
            # Last compressing stage (D) PR settings
            for comp_no in range(self.num_comp_list[0] + self.num_comp_list[1] + self.num_comp_list[2] + 1, self.num_compressors):
                self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{comp_no}\Input\\PRATIO").Value = PR4
            
            # Valve pressure for stage (except D) settings    
            for valve_no in range(self.num_comp_list[2]-1, 0, -1):
                if valve_no == self.num_comp_list[2] - 1:
                    globals()[f"V{valve_no}_pres"] = globals()[f"C{self.num_comp_list[0]}_pres"] * self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{self.num_comp_list[0] + self.num_comp_list[1] + 1}\\Input\\PRATIO").Value
                else:
                    globals()[f"V{valve_no}_pres"] = globals()[f"V{valve_no + 1}_pres"] * self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{self.num_comp_list[0] + self.num_comp_list[1] + self.num_comp_list[2] - valve_no}\\Input\\PRATIO").Value
                self.aspen.Tree.FindNode(f"\\Data\\Blocks\\V-{valve_no}\\Input\\P_OUT").Value = globals()[f"V{valve_no}_pres"]
            
            # Valve pressure for stage (D) settings    
            for valve_no in range(self.num_comp_list[2]+2, self.num_comp_list[2] + self.num_comp_list[3] + 1):
                if valve_no == self.num_comp_list[2] + 2:
                    globals()[f"V{valve_no}_pres"] = X[0][1] * self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{self.num_comp_list[0] + self.num_comp_list[1] + self.num_comp_list[2] + 1}\\Input\\PRATIO").Value 
                else:
                    globals()[f"V{valve_no}_pres"] = globals()[f"V{valve_no- 1}_pres"] * \
                        self.aspen.Tree.FindNode(f"\\Data\\Blocks\\C-{self.num_comp_list[0] + self.num_comp_list[1] + valve_no - 1}\\Input\\PRATIO").Value
                self.aspen.Tree.FindNode(f"\\Data\\Blocks\\V-{valve_no}\\Input\\P_OUT").Value = globals()[f"V{valve_no}_pres"]
        
        # Closed-loop
        elif self.simul_case == "Closed_loop":
            for valve_no in range(1, self.num_comp_list[-1]+1):
                if valve_no == 1:
                    globals()[f"V{valve_no}_pres"] = X[0][1].item()
                else:
                    globals()[f"V{valve_no}_pres"] = globals()[f"V{valve_no-1}_pres"] * X[0][self.num_comp_list[0]+valve_no-1]
                    self.aspen.Tree.FindNode(f"\\Data\\Blocks\\V-{valve_no}\\Input\\P_OUT").Value = globals()[f"V{valve_no}_pres"]
        
        # Run Aspen Plus simulation
        time.sleep(2)
        self.aspen.Engine.Run2()
        time.sleep(2)
        
        trial = 0
        while trial <= 1:
            # Simulation status
            status = self.check_convergence()
            
            # Default error status
            errflag = False
            errflag_predefined = False
            
            errflag_power = False
            errflag_duty = False
            errflag_H1 = False
            errflag_massflow = False
            
            if status == "Error":
                errflag = True
                errflag_predefined = True
            
            # Utility consumption
            if self.simul_case == "Open_loop":
                # Compressor power
                utility_comp_power_dict = {}
                for comp_no, node_power in enumerate(Y_nodes["Power"]):
                    globals()[f"C{comp_no+1}_power"] = self.aspen.Tree.FindNode(node_power).Value    # (kW)
                    
                    if globals()[f"C{comp_no+1}_power"] == None or 0:
                        errflag_power = True
                        errflag = True
                        break
                    else:
                        utility_comp_power_dict[f"C{comp_no+1}_power"] = globals()[f"C{comp_no+1}_power"]
                        
                # Cooler duty
                utility_cooler_duty_dict = {}
                for cooler_no, node_duty in enumerate(Y_nodes["Cooler_duty"]):
                    
                    if self.aspen.Tree.FindNode(node_duty).Value == None or self.aspen.Tree.FindNode(node_duty).Value > 0:
                        errflag_duty = True
                        errflag = True
                        break
                    else:
                        globals()[f"K{cooler_no+1}_duty"] = -1 * self.aspen.Tree.FindNode(node_duty).Value  # (cal/sec)
                        utility_cooler_duty_dict[f"K{cooler_no+1}_duty"] = globals()[f"K{cooler_no+1}_duty"] * 36 * 10**(-7) # (Gcal/hr) 
                
                # Heat exchanger LMTD
                H1_LMTD = self.aspen.Tree.FindNode(Y_nodes["H1_LMTD"][0]).Value    # (C)
                if H1_LMTD == None or 0:
                    errflag_H1 = True
                    errflag = True
                    
                # Heat exchanger duty
                H1_duty = self.aspen.Tree.FindNode(Y_nodes["H1_duty"][0]).Value    # (cal/sec)
                if H1_duty == None or 0:
                    if errflag_H1:
                        pass
                    else:
                        errflag_H1 = True
                        errflag = True
                
            elif self.simul_case == "Open&Closed_loop":
                # Compressor power
                utility_comp_power_dict = {}
                for comp_no, node_power in enumerate(Y_nodes["Power"]):
                    globals()[f"C{comp_no+1}_power"] = self.aspen.Tree.FindNode(node_power).Value # (kW)
                    
                    if globals()[f"C{comp_no+1}_power"] == None or 0:
                        errflag_power = True
                        errflag = True
                        break
                    else:
                        utility_comp_power_dict[f"C{comp_no+1}_power"] = globals()[f"C{comp_no+1}_power"]
                
                # Cooler duty
                utility_cooler_duty_dict = {}
                for cooler_no, node_duty in enumerate(Y_nodes["Cooler_duty"]):
                    
                    if self.aspen.Tree.FindNode(node_duty).Value == None or self.aspen.Tree.FindNode(node_duty).Value > 0:
                        errflag_duty = True
                        errflag = True
                        break
                    else:
                        globals()[f"K{cooler_no+1}_duty"] = -1 * self.aspen.Tree.FindNode(node_duty).Value   # (cal/sec)
                        utility_cooler_duty_dict[f"K{cooler_no+1}_duty"] = globals()[f"K{cooler_no+1}_duty"] * 36 * 10**(-7) # (Gcal/hr)
            
    
                # Heat exchanger LMTD
                H1_LMTD = self.aspen.Tree.FindNode(Y_nodes["H1_LMTD"][0]).Value # (C)
                if H1_LMTD == None or 0:
                    errflag_H1 = True
                    errflag = True

                # Heat exchanger duty
                H1_duty = self.aspen.Tree.FindNode(Y_nodes["H1_duty"][0]).Value    # (cal/sec)
                if H1_duty == None or 0:
                    if errflag_H1:
                        pass
                    else:
                        errflag_H1 = True
                        errflag = True
                                        
            elif self.simul_case == "Closed_loop":
                # Compressor power
                utility_comp_power_dict = {}
                for comp_no, node_power in enumerate(Y_nodes["Power"]):
                    globals()[f"C{comp_no+1}_power"] = self.aspen.Tree.FindNode(node_power).Value    # (kW)
                    
                    if globals()[f"C{comp_no+1}_power"] == None or 0:
                        errflag_power = True
                        errflag = True
                        break
                    else:
                        utility_comp_power_dict[f"C{comp_no+1}_power"] = globals()[f"C{comp_no+1}_power"]
                        
                # Cooler duty
                utility_cooler_duty_dict = {}
                for cooler_no, node_duty in enumerate(Y_nodes["Cooler_duty"]):
                    if self.aspen.Tree.FindNode(node_duty).Value == None or self.aspen.Tree.FindNode(node_duty).Value > 0:
                        errflag_duty = True
                        errflag = True
                        break
                    else:
                        globals()[f"K{cooler_no+1}_duty"] = -1 * self.aspen.Tree.FindNode(node_duty).Value    # (Gcal/hr)
                        utility_cooler_duty_dict[f"K{cooler_no+1}_duty"] = globals()[f"K{cooler_no+1}_duty"] # (Gcal/hr)

                # Heat exchanger LMTD
                H1_LMTD = self.aspen.Tree.FindNode(Y_nodes["H1_LMTD"][0]).Value # (C)
                if H1_LMTD == None or 0:
                    errflag_H1 = True
                    errflag = True

                # Heat exchanger duty
                H1_duty = self.aspen.Tree.FindNode(Y_nodes["H1_duty"][0]).Value    # (Gcal/hr)
                if H1_duty == None or 0:
                    if errflag_H1:
                        pass
                    else:
                        errflag_H1 = True
                        errflag = True        
                
            # Product mass flow rate
            prod_massflow = self.aspen.Tree.FindNode(Y_nodes["massflow"][0]).Value    # (kg/hr)
            if prod_massflow == 0:
                errflag_massflow = True
                errflag = True
            
            # Categorize the simulation results as error for the error cases
            if errflag:
                cost_dict = {}
                extrapolation_dict = {}
                cost_breakdown_dict = {}
                status = "Error"
                levelized_cost = 10e+50 + np.random.uniform(0, 1e+10)
                levelized_cost_turton = 10e+50 + np.random.uniform(0, 1e+10)
                
                if errflag_predefined:
                    pass
                
                else:
                    # Compressor power error
                    if errflag_power:
                        print(f"Iteration {self.history['iter']}: Aspen Plus simulation had compressor errors.", file = self.print_log)
                        
                    # Cooler duty error
                    elif errflag_duty:
                        print(f"Iteration {self.history['iter']} : Aspen Plus Simulation had cooler errors.", file = self.print_log)
                        
                    # H-1 heat exchanger error
                    elif errflag_H1:
                        print(f"Iteration {self.history['iter']} : Aspen Plus Simulation had H-1 errors.", file = self.print_log)
                        
                    # Product mass flow rate error
                    elif errflag_massflow:
                        print(f"Iteration {self.history['iter']} : Aspen Plus Simulation had errors in liquefying prod stream (Product had zero flow rate).", file = self.print_log)


                if trial == 0: 
                    if self.simul_case == "Open&Closed_loop":
                        flow_rate_multiplication_factor = self.aspen.Tree.FindNode("\\Data\\Blocks\\MULT\\Output\\R_FACTOR").Value
                        self.aspen.Tree.FindNode(f"\\Data\\Streams\\REFRIG\\Input\\FLOW\\MIXED\\{self.refrig}").Value = \
                            self.aspen.Tree.FindNode(f"\\Data\\Streams\\REFRIG\\Input\\FLOW\\MIXED\\{self.refrig}").Value * flow_rate_multiplication_factor
                        self.aspen.Reinit()
                        time.sleep(1)
                        self.aspen.Engine.Run2()
                        time.sleep(1)
                    trial += 1
                    continue 
                else:
                    break
            else:
                break 

        REFRIG_TPD = self.aspen.Tree.FindNode(f"\\Data\\Streams\\REFRIG\\Input\\FLOW\\MIXED\\{self.refrig}").Value

        # No error in calculation
        if errflag is False:
            cost_dict = {}
            extrapolation_dict = {}
            cost_breakdown_dict = {}

            if self.simul_case == "Open_loop":
                
                for cooler_no, node_cooler in enumerate(Y_nodes["Cooler_hot_in"]):
                    globals()[f"K{cooler_no+1}_hot_in"] = self.aspen.Tree.FindNode(node_cooler).Value    # (C)
                    
                for cooler_no, node_cooler in enumerate(Y_nodes["Cooler_pres"]):
                    globals()[f"K{cooler_no+1}_pres"] = self.aspen.Tree.FindNode(node_cooler).Value - 1.01325    # (barg)
                    
                for vessel_no, node_vessel in enumerate(Y_nodes["Vessel_inlet_liq_vol"]):
                    globals()[f"D{vessel_no+1}_inlet_liq_vol"] = self.aspen.Tree.FindNode(node_vessel).Value    # (L/min)
                    
                for vessel_no, node_vessel in enumerate(Y_nodes["Vessel_pres"]):
                    globals()[f"D{vessel_no+1}_pres"] = self.aspen.Tree.FindNode(node_vessel).Value - 1.01325    # (barg)

            elif self.simul_case == "Open&Closed_loop":
                
                for cooler_no, node_cooler in enumerate(Y_nodes["Cooler_hot_in"]):
                    globals()["K{}_hot_in".format(str(cooler_no + 1))] = self.aspen.Tree.FindNode(node_cooler).Value  # (C)
                
                for cooler_no, node_cooler in enumerate(Y_nodes["Cooler_pres"]):
                    globals()["K{}_pres".format(str(cooler_no + 1))] = self.aspen.Tree.FindNode(node_cooler).Value - 1.013 # (barg)
                            
                for vessel_no, node_vessel in enumerate(Y_nodes["Vessel_inlet_liq_vol"]):
                    globals()["D{}_inlet_liq_vol".format(str(vessel_no + 1))] = self.aspen.Tree.FindNode(node_vessel).Value # (L/min)

                for vessel_no, node_vessel in enumerate(Y_nodes["Vessel_pres"]):
                    globals()["D{}_pres".format(str(vessel_no + 1))] = self.aspen.Tree.FindNode(node_vessel).Value - 1.013 # (barg)                
            
            elif self.simul_case == "Closed_loop":

                for cooler_no, node_cooler in enumerate(Y_nodes["Cooler_hot_in"]):  
                    globals()["K{}_hot_in".format(str(cooler_no + 1))] = self.aspen.Tree.FindNode(node_cooler).Value # (C)
                   
                for cooler_no, node_cooler in enumerate(Y_nodes["Cooler_pres"]):
                    globals()["K{}_pres".format(str(cooler_no + 1))] = self.aspen.Tree.FindNode(node_cooler).Value - 1.013 # (barg)
                  
                for vessel_no, node_vessel in enumerate(Y_nodes["Vessel_inlet_liq_vol"]):
                    globals()["D{}_inlet_liq_vol".format(str(vessel_no + 1))] = self.aspen.Tree.FindNode(node_vessel).Value # (cum/hr)

                for vessel_no, node_vessel in enumerate(Y_nodes["Vessel_pres"]):
                    globals()["D{}_pres".format(str(vessel_no + 1))] = self.aspen.Tree.FindNode(node_vessel).Value - 1.013 # (barg)
                    
            # Compressor cost calculation, different materials for different configuration
            if self.simul_case == "Open_loop":
                # Compressors are CS
                for comp_no in range(1, self.num_compressors+1):
                    globals()[f"C{comp_no}_bare_module_cost"], globals()[f"C{comp_no}_extrapolation"] = \
                        self._comp_bare_module_cost(globals()[f"C{comp_no}_power"], "CS")
                        
                    cost_dict[f"C-{comp_no}"] = \
                        self._inflation_correction(globals()[f"C{comp_no}_bare_module_cost"],
                                                   cepci_target=self.CEPCI_2023,
                                                   cepci_ref=self.CEPCI_2001)
                    
                    extrapolation_dict[f"C-{comp_no}"] = globals()[f"C{comp_no}_extrapolation"]
                        
            elif self.simul_case == "Open&Closed_loop":
                # Compressors in zone 1 are SS
                for comp_no in range(1, self.num_comp_list[0]+1):
                    globals()[f"C{comp_no}_bare_module_cost"], globals()[f"C{comp_no}_extrapolation"] = \
                        self._comp_bare_module_cost(globals()[f"C{comp_no}_power"], "SS")
                    
                    cost_dict[f"C-{comp_no}"] = \
                            self._inflation_correction(globals()[f"C{comp_no}_bare_module_cost"],
                                                       cepci_target = self.CEPCI_2023,
                                                       cepci_ref = self.CEPCI_2001)
                            
                    extrapolation_dict[f"C-{comp_no}"] = globals()[f"C{comp_no}_extrapolation"]
                
                # Compressors in other zones are CS
                for comp_no in range(self.num_comp_list[0]+1, self.num_compressors+1):
                    globals()[f"C{comp_no}_bare_module_cost"], globals()[f"C{comp_no}_extrapolation"] = \
                        self._comp_bare_module_cost(globals()[f"C{comp_no}_power"], "CS")                    
            
                    cost_dict[f"C-{comp_no}"] = \
                            self._inflation_correction(globals()[f"C{comp_no}_bare_module_cost"],
                                                       cepci_target = self.CEPCI_2023,
                                                       cepci_ref = self.CEPCI_2001)
                            
                    extrapolation_dict[f"C-{comp_no}"] = globals()[f"C{comp_no}_extrapolation"]
 
            elif self.simul_case == "Closed_loop":
                # Compressors in zone 1 are SS
                for comp_no in range(1, self.num_comp_list[0]+1):
                    globals()[f"C{comp_no}_bare_module_cost"], globals()[f"C{comp_no}_extrapolation"] = \
                        self._comp_bare_module_cost(globals()[f"C{comp_no}_power"], "SS")
                        
                    cost_dict[f"C-{comp_no}"] = \
                        self._inflation_correction(globals()[f"C{comp_no}_bare_module_cost"],
                                                   cepci_target=self.CEPCI_2023,
                                                   cepci_ref=self.CEPCI_2001)
                    
                    extrapolation_dict[f"C-{comp_no}"] = globals()[f"C{comp_no}_extrapolation"]
                
                # Compressors in zone 2 are CS
                for comp_no in range(self.num_comp_list[0]+1, self.num_compressors+1):
                    globals()[f"C{comp_no}_bare_module_cost"], globals()[f"C{comp_no}_extrapolation"] = \
                        self._comp_bare_module_cost(globals()[f"C{comp_no}_power"], "CS")
                        
                    cost_dict[f"C-{comp_no}"] = \
                        self._inflation_correction(globals()[f"C{comp_no}_bare_module_cost"],
                                                   cepci_target=self.CEPCI_2023,
                                                   cepci_ref=self.CEPCI_2001)
                    
                    extrapolation_dict[f"C-{comp_no}"] = globals()[f"C{comp_no}_extrapolation"]            
            
            # Drive cost calculation
            for comp_no in range(1, self.num_compressors+1):
                globals()[f"C{comp_no}_drive_bare_module_cost"], globals()[f"C{comp_no}_drive_extrapolation"] = \
                    self._drive_bare_module_cost(globals()[f"C{comp_no}_power"] / self.mechanical_efficiency)
                    
                cost_dict[f"C-{comp_no}_drive"] = \
                    self._inflation_correction(globals()[f"C{comp_no}_drive_bare_module_cost"],
                                               cepci_target=self.CEPCI_2023,
                                               cepci_ref=self.CEPCI_2001)
                    
                extrapolation_dict[f"C-{comp_no}_drive"] = globals()[f"C{comp_no}_drive_extrapolation"]
            
            # Cooler cost calculation, different materials for different configuration
            if self.simul_case == "Open_loop":
                # Coolers except the last one are shell-and-tube type CS
                # The last cooler is plate type SS
                # Assume countercurrent, overall heat transfer coefficients are all the same
                for cooler_no in range(1, self.num_coolers+1):
                    hot_in = globals()[f"K{cooler_no}_hot_in"]
                    LMTD = self._lmtd_countercurrent(hot_in, hot_out=40, cold_in=35, cold_out=40)
                    
                    if cooler_no < self.num_coolers:
                        U = self.ST_U    # (cal/sec-sqcm-K)
                        material = "CS shell/CS tube"
                        
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1e-4)    # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] = \
                            self._cooler_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"], 
                                                          globals()[f"K{cooler_no}_pres"], material)
                                                          
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                       cepci_target=self.CEPCI_2023,
                                                       cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]
                            
                    else: 
                        U = self.plate_cooler_U[f"Open_loop_carbon_clean_comp{self.comp_type}_cooler{str(self.num_coolers)}_vessel{str(self.num_vessels)}"]
                        
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1e-4)    # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] = \
                            self._HE_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"])
                              
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                       cepci_target=self.CEPCI_2023,
                                                       cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]
                            
            elif self.simul_case == "Open&Closed_loop":
                # Coolers in zone 1 are shell-and-tube type SS
                # Coolers in zone 2 are shell-and-tube type CS
                # Coolers in zone 3 are plate type SS
                # Coolers in zone 4 are shell-and-tube type CS, except the last cooler is plate type SS
                # Assume countercurrent, overall heat transfer coefficients are all the same                
                for cooler_no in range(1, self.num_coolers + 1):
                    
                    hot_in = globals()[f"K{cooler_no}_hot_in"]
                    LMTD = self._lmtd_countercurrent(hot_in, hot_out=40, cold_in=35, cold_out=40)
                    
                    if cooler_no <= self.num_comp_list[0]:
                        U = self.ST_U 
                        material = "SS shell/SS tube"
                        
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1e-4) # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] = \
                            self._cooler_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"], 
                                                          globals()[f"K{cooler_no}_pres"], material)
                        
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                       cepci_target=self.CEPCI_2023,
                                                       cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]
                    
                    elif cooler_no > self.num_comp_list[0] and cooler_no <= self.num_comp_list[0] + self.num_comp_list[1]:
                        U = self.ST_U
                        material = "CS shell/CS tube"
                                           
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1e-4) # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] =\
                            self._cooler_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"],
                                                          globals()[f"K{cooler_no}_pres"], material)
                            
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                       cepci_target=self.CEPCI_2023,
                                                       cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]                            

                    elif cooler_no  > self.num_comp_list[0] + self.num_comp_list[1] and cooler_no <= self.num_comp_list[0] + self.num_comp_list[1] + self.num_comp_list[2]:
                        U = self.plate_cooler_U[f"Open&Closed_loop_carbon_clean_comp{self.comp_type}_cooler{self.num_coolers}_vessel{self.num_vessels}"]
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U /LMTD * (1e-4) # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] =\
                            self._HE_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"])
                        
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                       cepci_target=self.CEPCI_2023,
                                                       cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]   
                    
                    elif cooler_no == self.num_coolers:
                        U = self.plate_cooler_U[f"Open&Closed_loop_carbon_clean_comp{self.comp_type}_cooler{self.num_coolers}_vessel{self.num_vessels}"]
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U /LMTD * (1e-4) # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] =\
                            self._HE_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"])
                        
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                       cepci_target=self.CEPCI_2023,
                                                       cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]             
                                                                          
                    else:
                        U = self.ST_U
                        material = "CS shell/CS tube"
                                           
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1e-4) # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] =\
                            self._cooler_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"],
                                                          globals()[f"K{cooler_no}_pres"], material)
                            
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                       cepci_target=self.CEPCI_2023,
                                                       cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]                       
                                             
            
            elif self.simul_case == "Closed_loop":
                # Coolers for CO2 are shell-and-tube type SS
                # Coolers for refrigerant are shell-and-tube type CS (except the last one)
                # The very last cooler in the refrigerant cycle is plate type SS
                # Assume countercurrent, overall heat transfer coefficients are all the same
                for cooler_no in range(1, self.num_coolers+1):
                    hot_in = globals()[f"K{cooler_no}_hot_in"]
                    LMTD = self._lmtd_countercurrent(hot_in, hot_out=40, cold_in=35, cold_out=40)
                    if cooler_no <= self.num_comp_list[0]:
                        U = self.ST_U
                        material = "SS shell/SS tube"
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1000/36)    # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] = \
                            self._cooler_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"], 
                                                            globals()[f"K{cooler_no}_pres"], material)
                                                            
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                        cepci_target=self.CEPCI_2023,
                                                        cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]
                            
                    elif cooler_no == self.num_coolers:
                        U = self.plate_cooler_U[f"Closed_loop_carbon_clean_comp{self.comp_type}_cooler{str(self.num_coolers)}_vessel{str(self.num_vessels)}"]
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1000/36)    # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] = \
                            self._HE_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"])
                                
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                        cepci_target=self.CEPCI_2023,
                                                        cepci_ref=self.CEPCI_2001)
                            
                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]
                        
                    else:
                        U = self.ST_U
                        material = "CS shell/CS tube"
                        globals()[f"K{cooler_no}_area"] = globals()[f"K{cooler_no}_duty"] / U / LMTD * (1000/36)    # (sqm)
                        globals()[f"K{cooler_no}_bare_module_cost"], globals()[f"K{cooler_no}_extrapolation"] = \
                            self._cooler_bare_module_cost(self.sizing_margin * globals()[f"K{cooler_no}_area"], 
                                                            globals()[f"K{cooler_no}_pres"], material)
                                                            
                        cost_dict[f"K-{cooler_no}"] = \
                            self._inflation_correction(globals()[f"K{cooler_no}_bare_module_cost"],
                                                        cepci_target=self.CEPCI_2023,
                                                        cepci_ref=self.CEPCI_2001)

                        extrapolation_dict[f"K-{cooler_no}"] = globals()[f"K{cooler_no}_extrapolation"]
        
            # Heat exchanger cost calculation
            if self.simul_case == "Open_loop":
                U = self.plate_MSHE_U[f"Open_loop_carbon_clean_comp{self.comp_type}_cooler{self.num_coolers}_vessel{self.num_vessels}"]    # (cal/sec-sqcm-K)
                LMTD = H1_LMTD
                
                H1_area = H1_duty / U / LMTD * (1e-4)
                H1_bare_module_cost, H1_extrapolation = self._MSHE_bare_module_cost(self.sizing_margin * H1_area)
                
                cost_dict["H-1"] = \
                    self._inflation_correction(H1_bare_module_cost, cepci_target=self.CEPCI_2023, cepci_ref=self.CEPCI_2001)
                    
                extrapolation_dict["H-1"] = H1_extrapolation
                
            elif self.simul_case == "Open&Closed_loop":
                U = self.plate_MSHE_U[f"Open&Closed_loop_carbon_clean_comp{self.comp_type}_cooler{self.num_coolers}_vessel{self.num_vessels}_refrig_{self.refrig}"]
                LMTD = H1_LMTD
                
                H1_area = H1_duty / U / LMTD * (1e-4)
                H1_bare_module_cost, H1_extrapolation = self._MSHE_bare_module_cost(self.sizing_margin * H1_area)

                cost_dict["H-1"] = \
                    self._inflation_correction(H1_bare_module_cost, cepci_target=self.CEPCI_2023, cepci_ref=self.CEPCI_2001)
                    
                extrapolation_dict["H-1"] = H1_extrapolation                
            
            elif self.simul_case == "Closed_loop":
                
                U = self.plate_MSHE_U[f"Closed_loop_carbon_clean_comp{self.comp_type}_cooler{str(self.num_coolers)}_vessel{str(self.num_vessels)}_refrig_{self.refrig}"]    # (cal/sec-sqcm-K)
                LMTD = H1_LMTD                
                H1_area = H1_duty / U / LMTD * (1000/36)    # (sqm)
                H1_bare_module_cost, H1_extrapolation = self._HE_bare_module_cost(self.sizing_margin * H1_area)
                
                cost_dict["H-1"] = \
                    self._inflation_correction(H1_bare_module_cost, cepci_target=self.CEPCI_2023, cepci_ref=self.CEPCI_2001)
                extrapolation_dict["H-1"] = H1_extrapolation               
                
            # Vessel and dryer cost calculation
            # The first one is dryer, and the others are vessel
            # Dryer is CS
            # Vessels for CO2 streams are SS in open-loop, open&closed-loop, and closed-loop
            # Vessels for refrigerant streams are CS in open&closed-loop and closed-loop
            
            if self.simul_case == "Open_loop":
                for vessel_no in range(1, self.num_vessels + 1):
                    if vessel_no == 1:
                        globals()[f"D{vessel_no}_bare_module_cost"], globals()[f"D{vessel_no}_extrapolation"] =\
                            self._dryer_bare_module_cost(self.sizing_margin * globals()[f"D{vessel_no}_inlet_liq_vol"] / 100)
                            
                    else:
                        globals()[f"D{vessel_no}_bare_module_cost"], globals()[f"D{vessel_no}_extrapolation"] =\
                            self._vessel_bare_module_cost(self.sizing_margin * globals()[f"D{vessel_no}_inlet_liq_vol"]/100,
                                                          globals()[f"D{vessel_no}_pres"],
                                                          "SS")
                    
                    cost_dict[f"D-{vessel_no}"] = self._inflation_correction(globals()[f"D{vessel_no}_bare_module_cost"],
                                                                             cepci_target = self.CEPCI_2023,
                                                                             cepci_ref = self.CEPCI_2001)
                    extrapolation_dict[f"D-{vessel_no}"] = globals()[f"D{vessel_no}_extrapolation"]
                
            elif self.simul_case == "Open&Closed_loop":
                for vessel_no in range(1, self.num_vessels + 1):
                    if vessel_no == 1:
                        globals()[f"D{vessel_no}_bare_module_cost"], globals()[f"D{vessel_no}_extrapolation"] =\
                            self._dryer_bare_module_cost(self.sizing_margin * globals()[f"D{vessel_no}_inlet_liq_vol"] / 100)
                    
                    elif vessel_no >=2 and vessel_no <= self.num_comp_list[2]+1:
                        globals()[f"D{vessel_no}_bare_module_cost"], globals()[f"D{vessel_no}_extrapolation"] =\
                            self._vessel_bare_module_cost(self.sizing_margin * globals()[f"D{vessel_no}_inlet_liq_vol"]/100,
                                                          globals()[f"D{vessel_no}_pres"],
                                                          "SS")
                    else:
                        globals()[f"D{vessel_no}_bare_module_cost"], globals()[f"D{vessel_no}_extrapolation"] =\
                            self._vessel_bare_module_cost(self.sizing_margin * globals()[f"D{vessel_no}_inlet_liq_vol"]/100,
                                                          globals()[f"D{vessel_no}_pres"],
                                                          "CS")
                            
                    cost_dict[f"D-{vessel_no}"] = self._inflation_correction(globals()[f"D{vessel_no}_bare_module_cost"],
                                                                             cepci_target = self.CEPCI_2023,
                                                                             cepci_ref = self.CEPCI_2001)
                    extrapolation_dict[f"D-{vessel_no}"] = globals()[f"D{vessel_no}_extrapolation"]
                        
            elif self.simul_case == "Closed_loop":
                for vessel_no in range(1, self.num_vessels + 1):
                    if vessel_no == 1:
                        globals()[f"D{vessel_no}_bare_module_cost"], globals()[f"D{vessel_no}_extrapolation"] =\
                            self._dryer_bare_module_cost(self.sizing_margin * globals()[f"D{vessel_no}_inlet_liq_vol"] / 6)
                    
                    else:
                        globals()[f"D{vessel_no}_bare_module_cost"], globals()[f"D{vessel_no}_extrapolation"] =\
                            self._vessel_bare_module_cost(self.sizing_margin * globals()[f"D{vessel_no}_inlet_liq_vol"] / 6,
                                                          globals()[f"D{vessel_no}_pres"],
                                                          "CS")            
                    
                    cost_dict[f"D-{vessel_no}"] = self._inflation_correction(globals()[f"D{vessel_no}_bare_module_cost"],
                                                                             cepci_target = self.CEPCI_2023,
                                                                             cepci_ref = self.CEPCI_2001)
                    extrapolation_dict[f"D-{vessel_no}"] = globals()[f"D{vessel_no}_extrapolation"]

            
            # CAPEX calculation
            CAPEX = np.sum(list(cost_dict.values()))
            
            # OPEX_TURTON calculation
            OPEX_TURTON = self._opex_turton(cost_dict, utility_comp_power_dict, utility_cooler_duty_dict)
            OPEX, C_OM_proportion, cooling_cost_proportion, electricity_cost_proportion = self._opex(cost_dict, utility_comp_power_dict, utility_cooler_duty_dict)
            
            # Levelized cost calculation
            CAPEX_annual = CAPEX * (self.interest_rate * (1+self.interest_rate)**self.plant_year) / \
                           ((1+self.interest_rate)**self.plant_year - 1)
                           
            total_cost_annual_turton = CAPEX_annual + OPEX_TURTON
            
            total_cost_annual = CAPEX_annual + OPEX
            
            # Cost breakdown
            CAPEX_annual_proportion = CAPEX_annual / total_cost_annual
            OPEX_annual_proportion = OPEX / total_cost_annual
            C_OM_annual_proportion = OPEX_annual_proportion * C_OM_proportion
            cooling_cost_annual_proportion = OPEX_annual_proportion * cooling_cost_proportion
            electricity_cost_annual_proportion = OPEX_annual_proportion * electricity_cost_proportion
            
            cost_breakdown_dict = {"CAPEX_annual_proportion" : CAPEX_annual_proportion,
                                   "C_OM_annual_proportion" : C_OM_annual_proportion,
                                   "cooling_cost_annual_proportion" : cooling_cost_annual_proportion,
                                   "electricity_cost_annual_proportion" : electricity_cost_annual_proportion}
            
            # Levelized cost
            levelized_cost_turton = total_cost_annual_turton / (self.feed_flowrate * 365)
            levelized_cost = total_cost_annual / (self.feed_flowrate * 365)
            
        # Saving results
        if status == "Converged":
            # Exergy efficiency calculation
            total_power = np.sum([globals()[f"C{comp_no+1}_power"] for comp_no in range(self.num_compressors)])
            exergy_efficiency = np.expand_dims([self._calc_exergy_efficiency(Y_nodes, total_power)], axis=0)
            
            self.history["X_sample"] = np.vstack((self.history["X_sample"], X.reshape(1, -1)))
            self.history["Y_sample"] = np.vstack((self.history["Y_sample"], np.expand_dims([levelized_cost], axis=0)))
            self.history["Exergy_efficiency"] = np.vstack((self.history["Exergy_efficiency"], exergy_efficiency))
            self.history_cost[f"cost_dict_iter_{self.history_cost['iter']}"] = cost_dict
            self.history_extrapolation[f"extrapoation_dict_iter_{self.history_cost['iter']}"] = extrapolation_dict
            
            self.history["Y_sample_turton"] = np.vstack((self.history["Y_sample_turton"], np.expand_dims([levelized_cost_turton], axis = 0)))
            self.history["REFRIG_TPD"] = np.vstack((self.history["REFRIG_TPD"], np.expand_dims([REFRIG_TPD], axis = 0)))

            self.history_cost_breakdown[f"cost_dict_iter_{self.history_cost['iter']}"] = cost_breakdown_dict

        elif status == "Warning":
            # Exergy efficiency calculation
            total_power = np.sum([globals()[f"C{comp_no+1}_power"] for comp_no in range(self.num_compressors)])
            exergy_efficiency = np.expand_dims([self._calc_exergy_efficiency(Y_nodes, total_power)], axis=0)
            
            self.history["X_sample"] = np.vstack((self.history["X_sample"], X.reshape(1, -1)))
            self.history["Y_sample"] = np.vstack((self.history["Y_sample"], np.expand_dims([levelized_cost], axis=0)))
            self.history["X_sample_warning"] = np.vstack((self.history["X_sample_warning"], X.reshape(1, -1)))
            self.history["Y_sample_warning"] = np.vstack((self.history["Y_sample_warning"], np.expand_dims([levelized_cost], axis=0)))
            self.history["Exergy_efficiency"] = np.vstack((self.history["Exergy_efficiency"], exergy_efficiency))
            self.history_cost[f"cost_dict_iter_{self.history_cost['iter']}"] = cost_dict
            self.history_extrapolation[f"extrapoation_dict_iter_{self.history_cost['iter']}"] = extrapolation_dict
            
            self.history["Y_sample_turton"] = np.vstack((self.history["Y_sample_turton"], np.expand_dims([levelized_cost_turton], axis=0)))
            self.history["Y_sample_warning_turton"] = np.vstack((self.history["Y_sample_warning_turton"], np.expand_dims([levelized_cost_turton], axis=0)))
            self.history["REFRIG_TPD"] = np.vstack((self.history["REFRIG_TPD"], np.expand_dims([REFRIG_TPD], axis = 0)))

            self.history_cost_breakdown[f"cost_dict_iter_{self.history_cost['iter']}"] = cost_breakdown_dict

        elif status == "Error":
            # Set exergy efficiency as zero
            self.history["X_sample"] = np.vstack((self.history["X_sample"], X.reshape(1, -1)))
            self.history["Y_sample"] = np.vstack((self.history["Y_sample"], np.expand_dims([levelized_cost], axis=0)))
            self.history["X_sample_error"] = np.vstack((self.history["X_sample_error"], X.reshape(1, -1)))
            self.history["Y_sample_error"] = np.vstack((self.history["Y_sample_error"], np.expand_dims([levelized_cost], axis=0)))
            self.history["Exergy_efficiency"] = np.vstack((self.history["Exergy_efficiency"], 0.))
            self.history_cost[f"cost_dict_iter_{self.history_cost['iter']}"] = cost_dict
            self.history_extrapolation[f"extrapoation_dict_iter_{self.history_cost['iter']}"] = extrapolation_dict
            
            self.history["Y_sample_turton"] = np.vstack((self.history["Y_sample_turton"], np.expand_dims([levelized_cost_turton], axis=0)))
            self.history["Y_sample_error_turton"] = np.vstack((self.history["Y_sample_error_turton"], np.expand_dims([levelized_cost_turton], axis=0)))
            self.history["REFRIG_TPD"] = np.vstack((self.history["REFRIG_TPD"], np.expand_dims([REFRIG_TPD], axis = 0)))

            self.history_cost_breakdown[f"cost_dict_iter_{self.history_cost['iter']}"] = cost_breakdown_dict

        # Purge all Aspen Plus simulation results
        time.sleep(1)
        if self.history["iter"] % 200 == 0 and self.history["iter"] != self.n_iter:
            self.aspen.Close()
            
            self.aspen = win32.Dispatch("Apwn.Document")
            self.aspen.InitFromArchive2(os.path.abspath(self.simul_file))
            self.aspen.Visible = False
            self.aspen.SuppressDialogs = True
        else:
            self.aspen.Reinit()
            
        time.sleep(1)
        
        # Plot figures for each iteration
        fig = self._sample_plot(self.history["X_sample"], self.history["Y_sample"])
        fig.savefig(f"temp_saved_fig/{self.temp_save_filename}/{self.history['iter']:03d}.png")
        plt.close()
        
        # Save results for each iteration
        with open(f"temp_saved_results/{self.temp_save_filename}.pkl", "wb") as f:
            dump(self.history, f)
            
        with open(f"temp_saved_results/cost_dict_{self.temp_save_filename}.pkl", "wb") as f:
            dump(self.history_cost, f)
            
        with open(f"temp_saved_results/extrapolation_dict_{self.temp_save_filename}.pkl", "wb") as f:
            dump(self.history_extrapolation, f)

        with open(f"temp_saved_results/cost_breakdown_dict_{self.temp_save_filename}.pkl", "wb") as f:
            dump(self.history_cost_breakdown, f)

        # Add 1 for each iteration
        self.history["iter"] += 1
        self.history_cost["iter"] += 1
        self.history_extrapolation["iter"] += 1
        self.history_cost_breakdown["iter"] += 1
        
        return levelized_cost
    
    
    def _comp_bare_module_cost(self, fluid_power, material):
        """
        Compressor bare module cost at the reference CEPCI (397).

        Parameters
        ----------
        fluid_power : float
            Fluid power of the compressor calculated from Aspen Plus.
        material : string
            Material of construction (CS, SS, Ni alloy).

        Raises
        ------
        ValueError
            If material of construction is not one of the followings:
                CS, SS, Ni alloy.

        Returns
        -------
        bare_module_cost : float
            Bare module cost of the compressor at the reference CEPCI.
        is_extrapolation : boolean
            Check if the extrapolation is used for the bare module cost calculation.

        """
        is_extrapolation = False
        
        # Extrapolation for the outside-range fluid power of purchased cost calculation
        if fluid_power < 18:    # (kW)
            is_extrapolation = True
        elif fluid_power > 3000:
            is_extrapolation = True
            
        # Different compressor type for the fluid power
        if fluid_power < 50:    # (kW)
            # Rotary compressor
            K1 = 5.0355
            K2 = -1.8002
            K3 = 0.8253
        else:
            # Centrifugal compressor
            K1 = 2.2897
            K2 = 1.3604
            K3 = -0.1027
            
        # Different bare module factor for different material of construction
        if material == "CS":
            if fluid_power < 50:
                bare_module_factor = 2.4
            else:
                bare_module_factor = 2.75
        elif material == "SS":
            if fluid_power < 50:
                bare_module_factor = 5
            else:
                bare_module_factor = 5.75
        elif material == "Ni alloy":
            if fluid_power < 50:
                bare_module_factor = 9.85
            else:
                bare_module_factor = 11.45
        else:
            raise ValueError("Bare module factor should be one of the followings: CS, SS, Ni alloy")
        
        # Reference purchased cost calculation (CEPCE=397)
        log10_power = np.log10(fluid_power)
        log10_purchased_cost_ref = K1 + K2 * log10_power + K3 * log10_power**2
        purchased_cost_ref = 10 ** log10_purchased_cost_ref
        
        # Bare module cost calculation
        bare_module_cost = purchased_cost_ref * bare_module_factor
        
        return bare_module_cost, is_extrapolation
    
    
    def _drive_bare_module_cost(self, shaft_power):
        """
        Drive bare module cost at the reference CEPCI (397).

        Parameters
        ----------
        shaft_power : float
            Shaft power of the drive calculated from Aspen Plus.

        Returns
        -------
        bare_module_cost : float
            Bare module cost of the drive at the reference CEPCI.
        is_extrapolation : boolean
            Check if the extrapolation is used for the bare module cost calculation.

        """
        is_extrapolation = False
        
        # Extrapolation for the outside-range shaft power of purchase cost calculation
        if shaft_power < 75:    # (kW)
            is_extrapolation = True
        elif shaft_power > 2600:    # (kW)
            is_extrapolation = True
            
        # Electric drive (explosion proof)
        K1 = 2.4604
        K2 = 1.4191
        K3 = -0.1798
        
        # Bare module factor for electric drive (explosion proof)
        bare_module_factor = 1.5
        
        # Reference purchased cost calculation (CEPCI=397)
        log10_power = np.log10(shaft_power)
        log10_purchased_cost_ref = K1 + K2 * log10_power + K3 * log10_power**2
        purchased_cost_ref = 10 ** log10_purchased_cost_ref
        
        # Bare module cost calculation
        bare_module_cost = purchased_cost_ref * bare_module_factor
        
        return bare_module_cost, is_extrapolation
    
    
    def _cooler_bare_module_cost(self, area, pressure, material):
        """
        Cooler bare module cost at the reference CEPCI (397).
        Assumes shell-and-tube type heat exchanger.

        Parameters
        ----------
        area : float
            Heat transfer area of the cooler calculated from the Aspen Plus simulation.
        pressure : float
            Operating pressure of the cooler.
        material : string
            Material of construction (CS shell/CS tube, SS shell/SS tube).

        Raises
        ------
        ValueError
            If material of construction is not one of the followings:
                CS shell/CS tube, SS shell/SS tube.

        Returns
        -------
        bare_module_cost : float
            Bare module cost of cooler at the reference CEPCI.
        is_extrapolation : boolean
            Check if the extrapolation is used for the bare module cost calculation.

        """
        is_extrapolation = False
        
        # Extrapolation for the outside-range area of purchased cost calculation
        if area < 10:    # (sqm)
            is_extrapolation = True
        elif area > 1000:    # (sqm)
            is_extrapolation = True
            
        # Fixed-tube heat exchanger
        K1 = 4.3247
        K2 = -0.303
        K3 = 0.1634
        
        # Pressure factor calculation
        if pressure < 5:    # (barg)
            pressure_factor = 1
        else:
            C1 = 0.03881
            C2 = -0.11272
            C3 = 0.08183
            
            log10_pressure = np.log10(pressure)
            pressure_factor = 10 ** (C1 + C2 * log10_pressure + C3 * log10_pressure**2)
            
        # # Different bare module factor for different material of construction
        if material == "CS shell/CS tube":
            material_factor = 1
        elif material == "SS shell/SS tube":
            material_factor = 2.75
        else:
            raise ValueError("Bare module factor should be one of the followings: CS shell/CS tube, SS shell/SS tube.")
            
        # Reference purchased cost calculation (CEPCI=397)
        log10_area = np.log10(area)
        log10_purchased_cost_ref = K1 + K2 * log10_area + K3 * log10_area**2
        purchased_cost_ref = 10 ** log10_purchased_cost_ref
            
        # Bare module cost calculation (fixed tube sheet)
        B1 = 1.63
        B2 = 1.66
        bare_module_cost = purchased_cost_ref * (B1 + B2 * material_factor * pressure_factor)
        
        return bare_module_cost, is_extrapolation
    
    
    def _MSHE_bare_module_cost(self, area):
        """
        Multistream heat exchanger bare module cost at the reference CEPCI (397).
        Assumes plate-fin type heat exchanger.

        Parameters
        ----------
        area : float
            Heat transfer area of the multistream heat exchanger calculated from the Aspen Plus simulation.

        Returns
        -------
        bare_module_cost : float
            Bare module cost of multistream heat exchanger at the reference CEPCI.
        is_extrapolation : boolean
            Check if the extrapolation is used for the bare module cost calculation.

        """
        is_extrapolation = False
        
        if area < 10:    # (sqm)
            is_extrapolation = True
        elif area > 1000:    # (sqm)
            is_extrapolation = True
            
        # Plate-fin heat exchanger for open-loop and open&closed-loop
        K1 = 4.6656
        K2 = -0.1557
        K3 = 0.1547
        
        # Pressure factor calculation
        pressure_factor = 1
        
        # Material factor calculation (Only SS)
        material_factor = 2.45
        
        # Reference purchased cost calculation (CEPCI=397)
        log10_area = np.log10(area)
        log10_purchased_cost_ref = K1 + K2 * log10_area + K3 * log10_area**2
        purchased_cost_ref = 10 ** log10_purchased_cost_ref
        
        # Bare module cost calculation (flat plate)
        B1 = 0.96
        B2 = 1.21
        bare_module_cost = purchased_cost_ref * (B1 + B2 * material_factor * pressure_factor)
        
        return bare_module_cost, is_extrapolation
    
    
    def _HE_bare_module_cost(self, area):
        """
        Heat exchanger bare module cost at the reference CEPCI (397).
        Assumes plate type heat exchanger.

        Parameters
        ----------
        area : float
            Heat transfer area of the heat exchanger calculated from the Aspen Plus simulation.

        Returns
        -------
        bare_module_cost : float
            Bare module cost of heat exchanger at the reference CEPCI.
        is_extrapolation : boolean
            Check if the extrapolation is used for the bare module cost calculation.

        """
        is_extrapolation = False
        
        if area < 10:    # (sqm)
            is_extrapolation = True
        elif area > 1000:    # (sqm)
            is_extrapolation = True
            
        # Plate-fin heat exchanger for open-loop and open&closed-loop
        K1 = 4.6656
        K2 = -0.1557
        K3 = 0.1547
        
        # Pressure factor calculation
        pressure_factor = 1
        
        # Material factor calculation (Only SS)
        material_factor = 2.45
        
        # Reference purchased cost calculation (CEPCI=397)
        log10_area = np.log10(area)
        log10_purchased_cost_ref = K1 + K2 * log10_area + K3 * log10_area**2
        purchased_cost_ref = 10 ** log10_purchased_cost_ref
        
        # Bare module cost calculation (flat plate)
        B1 = 0.96
        B2 = 1.21
        bare_module_cost = purchased_cost_ref * (B1 + B2 * material_factor * pressure_factor)
        
        return bare_module_cost, is_extrapolation
    
    
    def _vessel_bare_module_cost(self, volume, pressure, material):
        """
        Vessel bare module cost at the reference CEPCI (397).

        Parameters
        ----------
        volume : float
            Volume of the vessel calculated from the liquid flow rate flowing
            into the vessel at the Aspen Plus simulation.
        pressure : float
            Operating pressure of the vessel.
        material : string
            Material of construction (CS, SS).

        Raises
        ------
        ValueError
            If material of construction is not one of the followings:
                CS, SS.

        Returns
        -------
        bare_module_cost : float
            Bare module cost of vessel at the reference CEPCI.
        is_extrapolation : boolean
            Check if the extrapolation is used for the bare module cost calculation.

        """
        is_extrapolation = False
        
        # Extrapolation for the outside-range volume of purchased cost calculation
        if volume < 0.3:    # (cum)
            is_extrapolation = True
        elif volume > 520:    # (cum)
            is_extrapolation = True
            
        # Vertical vessel
        K1 = 3.4974
        K2 = 0.4485
        K3 = 0.1074
        
        # Assume 2:1 elliptical head vertical vessel with
        # H/D = 3, D = ( (6*volume) / (5*np.pi) )**(1/3)
        D = ((6 * volume) / (5 * np.pi))**(1/3)
        
        # Pressure factor calculation
        pressure_factor = ((pressure * D) / (2 * (850 - 0.6*pressure)) + 0.0018) / 0.0063
        if pressure_factor < 1:
            pressure_factor = 1
            
        # Different Material factor for different material of construction
        if material == "CS":
            material_factor = 1
        elif material == "SS":
            material_factor = 3.1
        else:
            raise ValueError("Bare module factor should be one of the followings: CS, SS")
            
        # Reference purchased cost calculation (CEPCI=397)
        log10_volume = np.log10(volume)
        log10_purchased_cost_ref = K1 + K2 * log10_volume + K3 * log10_volume**2
        purchased_cost_ref = 10 ** log10_purchased_cost_ref
        
        # Bare module cost calculation
        B1 = 2.25
        B2 = 1.82
        bare_module_cost = purchased_cost_ref * (B1 + B2 * material_factor * pressure_factor)
        
        return bare_module_cost, is_extrapolation
    
    
    def _dryer_bare_module_cost(self, volume):
        """
        Dryer bare module cost at the reference CEPCI (397).

        Parameters
        ----------
        volume : float
            Volume of the dryer calculated from the liquid flow rate flowing
            into the dryer at the Aspen Plus simulation.
            
        Returns
        -------
        bare_module_cost : float
            Bare module cost of dryer at the reference CEPCI.
        is_extrapolation : boolean
            Check if the extrapolation is used for the bare module cost calculation.

        """
        is_extrapolation = False
        
        # Assume 2:1 elliptical head vertical vessel with
        # H/D = 3, D = ( (6*volume) / (5*np.pi) )**(1/3)
        n = 1.6075
        D = ((6 * volume) / (5 * np.pi))**(1/3)
        
        # Surface area calculation: Cylindrical surface area + Ellipsoid surface area
        area = 3 * (D**2) * np.pi + 4 * np.pi * ((D**(2*n) +  (0.5 * D**2)**n + (0.5 * D**2)**n)/3)**(1/n)
        
        # Extrapolation for the outside-range area of pruchased cost calculation
        if area < 0.5:    # (sqm)
            is_extrapolation = True
        elif area > 50:    # (sqm)
            is_extrapolation = True
            
        # Drum
        K1 = 4.5472
        K2 = 0.2731
        K3 = 0.134
        
        # Bare module factor
        bare_module_factor = 1.6
        
        # Reference purchased cost calculation (CEPCI=397)
        log10_area = np.log10(area)
        log10_purchased_cost_ref = K1 + K2 * log10_area + K3 * log10_area**2
        purchased_cost_ref = 10 ** log10_purchased_cost_ref
        
        # Bare module cost calculation
        bare_module_cost = purchased_cost_ref * bare_module_factor
        
        return bare_module_cost, is_extrapolation
    
    
    def _lmtd_countercurrent(self, hot_in, hot_out=40, cold_in=35, cold_out=40):
        dT1 = hot_in - cold_out    # (C)
        dT2 = hot_out - cold_in    # (C)
        
        LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
        
        return LMTD
    
    
    def _inflation_correction(self, bare_module_cost_ref, cepci_target, cepci_ref=397):
        """
        Inflation correction based on CEPCI values.

        Parameters
        ----------
        bare_module_cost_ref : float
            Bare module cost at the reference CEPCI.
        cepci_target : int
            CEPCI value at the target calculating year.
        cepci_ref : int, optional
            CEPCI value at the reference calculating year (201). The default is 397.

        Returns
        -------
        bare_module_cost : TYPE
            DESCRIPTION.

        """
        bare_module_cost = bare_module_cost_ref * (cepci_target / cepci_ref)
        
        return bare_module_cost
    
    
    def _opex_turton(self, cost_dict, utility_comp_power_dict, utility_cooler_duty_dict):
        """
        Calculates OPEX of the plant.

        Parameters
        ----------
        cost_dict : dict
            Equipment costs for the CAPEX of the plant.
        utility_comp_power_dict : dict
            Compressor electricity consumption for each compressor.
        utility_cooler_duty_dict : dict
            Cooling duty consumption for each cooler.

        Returns
        -------
        OPEX : float
            OPEX of the plant.

        """
        # Fixed capital investment
        FCI= np.sum(list(cost_dict.values()))
        
        # Operating labor cost based on mean annual wage of chemical engineers in 2023, USA
        # U.S. Bureau of labor statistics (https://www.bls.gov/oes/2023/may/oes172041.htm)
        # Annual wage $122,910/yr (260 shifts/yr/person) -> $115,819/yr (245 shifts/yr/person)
        # Plant 365days/yr, 3 shifts/day -> 1095 shifts/yr -> 4.5 workers are needed
        n_workers = 4.5
        annual_wage = 115819
        
        P = 0    # Particulate solid processes
        N_np = len(cost_dict.keys()) - self.num_vessels - self.num_compressors    # Nonparticulate processes (compressors, heat exchangers, etc.)
        N_OL = (6.29 + 31.7 * P**2 + 0.23 * N_np) ** 0.5
        C_OL = n_workers * annual_wage * N_OL
        
        # Utility cost
        # Cooling water: $0.00245327/kWh
        # Electricity: $0.0775/kWh
        # Operating hour of the plant: 8000hrs/yr
        unit_cost_cooling = 0.00245327
        unit_cost_electricity = 0.0775
        
        unit_conversion_factor = 1163    # (Gcal/hr -> kWh/hr)
        operating_hour = 8000    # (hrs/yr)
        
        cooling_duty_required = np.sum(list(utility_cooler_duty_dict.values()))    # (Gcal/hr)
        cooling_cost = cooling_duty_required * unit_cost_cooling * unit_conversion_factor * operating_hour
        
        electricity_required = np.sum(list(utility_comp_power_dict.values()))    # (kW)
        electricity_cost = electricity_required * unit_cost_electricity * operating_hour
        
        C_UT = cooling_cost + electricity_cost
        
        # Waste treatment cost
        C_WT = 0
        
        # Raw material cost
        C_RM = 0
        
        # Depreciation = 0.1 FCI
        depreciation = 0.1 * FCI
        
        OPEX_TURTON = 0.180 * FCI + 2.73 * C_OL + 1.23 * (C_UT + C_WT + C_RM) + depreciation
        
        return OPEX_TURTON
    
    def _opex(self, cost_dict, utility_comp_power_dict, utility_cooler_duty_dict):
        """
        Calculates OPEX of the plant.

        Parameters
        ----------
        cost_dict : dict
            Equipment costs for the CAPEX of the plant.
        utility_comp_power_dict : dict
            Compressor electricity consumption for each compressor.
        utility_cooler_duty_dict : dict
            Cooling duty consumption for each cooler.

        Returns
        -------
        OPEX : float
            OPEX of the plant.

        """

        # Fixed capital investment
        FCI= np.sum(list(cost_dict.values()))  
        
        # O&M cost
        C_OM = FCI * 0.12
        
        # Utility cost
        # Cooling water: $0.00245327/kWh
        # Electricity: $0.0775/kWh
        # Operating hour of the plant: 8000hrs/yr
        unit_cost_cooling = 0.00245327
        unit_cost_electricity = 0.0775
        
        unit_conversion_factor = 1163    # (Gcal/hr -> kWh/hr)
        operating_hour = 8000    # (hrs/yr)
        
        cooling_duty_required = np.sum(list(utility_cooler_duty_dict.values()))    # (Gcal/hr)
        cooling_cost = cooling_duty_required * unit_cost_cooling * unit_conversion_factor * operating_hour
        
        electricity_required = np.sum(list(utility_comp_power_dict.values()))    # (kW)
        electricity_cost = electricity_required * unit_cost_electricity * operating_hour
        
        C_UT = cooling_cost + electricity_cost
        
        # Waste treatment cost
        C_WT = 0
        
        # Raw material cost
        C_RM = 0
        
        OPEX =  C_OM + C_UT + C_WT + C_RM 

        C_OM_proportion = C_OM / OPEX
        cooling_cost_proportion = cooling_cost / OPEX
        electricity_cost_proportion = electricity_cost / OPEX

        return OPEX, C_OM_proportion, cooling_cost_proportion, electricity_cost_proportion

    def _calc_exergy_efficiency(self, Y_nodes, total_power):
        """
        Calculates exergy efficiency of the process.

        Parameters
        ----------
        Y_nodes : dict
            Dictionary of nodes from each configuration.
        total_power : float
            Total power consumed in the process.

        Returns
        -------
        exergy_efficiency : float
            Exergy effiiciency of the process.

        """
        # CO2 mass flow rate
        feed_massflow_co2 = self.aspen.Tree.FindNode(Y_nodes["massflow_co2"][0]).Value / 3600    # (kg/s)
        prod_massflow_co2 = self.aspen.Tree.FindNode(Y_nodes["massflow_co2"][1]).Value / 3600    # (kg/s)
        
        # H2O mass flow rate
        feed_massflow_h2o = self.aspen.Tree.FindNode(Y_nodes["massflow_h2o"][0]).Value / 3600    # (kg/s)
        prod_massflow_h2o = self.aspen.Tree.FindNode(Y_nodes["massflow_h2o"][1]).Value / 3600    # (kg/s)
        
        # Physical exergy calculation
        feed_exergy_phys = self.aspen.Tree.FindNode(Y_nodes["exergy_phys"][0]).Value
        prod_exergy_phys = self.aspen.Tree.FindNode(Y_nodes["exergy_phys"][1]).Value
        
        # Chemical exergy calculation
        feed_exergy_chem = 0
        prod_exergy_chem = 0
        
        feed_molefrac_co2 = self.aspen.Tree.FindNode(Y_nodes["molefrac_co2"][0]).Value
        prod_molefrac_co2 = self.aspen.Tree.FindNode(Y_nodes["molefrac_co2"][1]).Value
        
        feed_exergy_chem += feed_massflow_co2 * \
                            (feed_molefrac_co2 * self.exergy_co2_gas_ref) + \
                            self.temperature_ref * self.gas_const / self.mw_co2 * feed_molefrac_co2 * np.log(feed_molefrac_co2)
        prod_exergy_chem += prod_massflow_co2 * \
                            (prod_molefrac_co2 * self.exergy_co2_gas_ref) + \
                            self.temperature_ref * self.gas_const / self.mw_co2 * prod_molefrac_co2 * np.log(prod_molefrac_co2)
                            
        feed_molefrac_h2o = self.aspen.Tree.FindNode(Y_nodes["molefrac_h2o"][0]).Value
        prod_molefrac_h2o = self.aspen.Tree.FindNode(Y_nodes["molefrac_h2o"][1]).Value
        
        feed_exergy_chem += feed_massflow_h2o * \
                            (feed_molefrac_h2o * self.exergy_h2o_gas_ref) + \
                            self.temperature_ref * self.gas_const / self.mw_h2o * feed_molefrac_h2o * np.log(feed_molefrac_h2o)
        prod_exergy_chem += prod_massflow_h2o * \
                            (prod_molefrac_h2o * self.exergy_h2o_liq_ref) + \
                            self.temperature_ref * self.gas_const / self.mw_h2o * prod_molefrac_h2o * np.log(prod_molefrac_h2o)
                            
        # Total exergy = Physical exergy + Chemical exergy
        feed_exergy = feed_exergy_phys + feed_exergy_chem
        prod_exergy = prod_exergy_phys + prod_exergy_chem
        
        # Exergy efficiency calculation
        exergy_efficiency = (prod_exergy - feed_exergy) / total_power
        
        return exergy_efficiency
    
    
    def _sample_plot(self, X_sample, Y_sample, to_plot="levelized_cost"):
        n_samples, n_vars = X_sample.shape
        
        fig_nrows = 2
        fig_ncols = n_vars//2 + n_vars%2
        fig, axes = plt.subplots(fig_nrows, fig_ncols, figsize=(4*fig_ncols, 2*fig_nrows))
        
        if n_samples == 1:
            X_now = X_sample[0, :]
            Y_now = Y_sample[0, :]
            
            for i in range(n_vars):
                axes[i%2, i//2].plot(X_now[i], Y_now[0], "o", color="#ff4d4d", markersize=3)
                axes[i%2, i//2].set_xlabel(f"{self.simul.X_names[i]}")
                axes[i%2, i//2].set_xlim(self.simul.bounds[i]['domain'])
                if to_plot == "levelized_cost":
                    axes[i%2, i//2].set_ylabel("Levelized cost ($/tCO2)")
                    axes[i%2, i//2].set_ylim([10, 30])
                elif to_plot == "exergy_efficiency":
                    axes[i%2, i//2].set_ylabel("Exergy efficiency")
                    axes[i%2, i//2].set_ylim([0, 1])
                axes[i%2, i//2].grid(linewidth=0.5)
                
            fig.tight_layout()
        
        else:
            X_prevs = X_sample[:-1, :]
            Y_prevs = Y_sample[:-1, :]
            
            X_now = X_sample[-1, :]
            Y_now = Y_sample[-1, :]
            
            for i in range(n_vars):
                axes[i%2, i//2].plot(X_prevs[:, i], Y_prevs[:, 0], "o", color="#555555", markersize=3)
                axes[i%2, i//2].plot(X_now[i], Y_now[0], "o", color="#ff4d4d", markersize=3)
                axes[i%2, i//2].set_xlabel(f"{self.simul.X_names[i]}")
                axes[i%2, i//2].set_xlim(self.simul.bounds[i]['domain'])
                if to_plot == "levelized_cost":
                    axes[i%2, i//2].set_ylabel("Levelized cost ($/tCO2)")
                    axes[i%2, i//2].set_ylim([10, 30])
                elif to_plot == "exergy_efficiency":
                    axes[i%2, i//2].set_ylabel("Exergy efficiency")
                    axes[i%2, i//2].set_ylim([0, 1])
                axes[i%2, i//2].grid(linewidth=0.5)
                
            fig.tight_layout()
        
        return fig