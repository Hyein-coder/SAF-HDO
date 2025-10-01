#%%
import os
from aspen_utils import AspenSim
from utils import CustomEvaluator
import GPy, GPyOpt
from GPyOpt.methods import ModularBayesianOptimization
import numpy as np
import datetime
import matplotlib.pyplot as plt

case_target = "a"
# past_path = None
past_path = r"D:\saf_hdo\results\a_20251001_165004.xlsx"
# aspen_path = "D:\\SAF-HDO py\\250926_pyrolysis_oil_CC_2.apw"
aspen_path = "D:/saf_hdo/aspen/250926_pyrolysis_oil_CC_rxn_index.apw"


sim = AspenSim(aspen_path, case_target=case_target) # , case_target='j'
save_dir = os.path.join(os.getcwd(), "results")
file_tag = case_target + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

cc = sim.get_carbon_number_composition(sim.prod_stream)
print(cc)

domain = [
    {'name': f'x{n_rxn}{i}', 'type': 'continuous', 'domain': (0, 1)} for n_rxn in sim.n_rxn for i in range(n_rxn)
]
r_now = sim.get_rxn_coefficients()

excel_path = os.path.join(save_dir, f"{file_tag}.xlsx")

var_names = []
for i, ris in enumerate(sim.rxn_indices):
    for ii, rxn_idx in enumerate(ris):
        var_names.append(f"R{i+1}_rxn{rxn_idx}")
var_names.append("OBJ")
for n in sim.carbon_number_to_component.keys():
    var_names.append(f"Prod_C{n}")

def hdo_prod_mse(x):
    sim.apply_rxn_coefficients(x)
    res = sim.get_carbon_number_composition(sim.prod_stream)
    mse = sum((sim.target[n] - res[n])**2 for n in sim.target.keys()) * 10
    print(mse)

    optional_res = np.array(list(res.values())).reshape(1, -1)
    return mse, optional_res
hdo_prod_mse(r_now)

#%%
evaluator = CustomEvaluator(hdo_prod_mse, excel_path, var_names)
objective = GPyOpt.core.task.SingleObjective(evaluator.evaluate)
r_init = np.array(r_now)

m52 = GPy.kern.Matern52(input_dim=r_init.shape[1], variance=1.0, lengthscale=1.0)

# Gaussian process regression model
model = GPyOpt.models.GPModel(kernel=m52, exact_feval=True,
                              optimize_restarts=25, verbose=False)

# Acquisition function: Expected improvement (EI)
domain_space = GPyOpt.Design_space(space=domain)

acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(domain_space)
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, domain_space, optimizer=acquisition_optimizer)
bo_evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# Set Bayesian optimization
if past_path is None:
    BO = ModularBayesianOptimization(model, domain_space, objective, acquisition, bo_evaluator, r_init)
else:
    past_data = evaluator.read_past_results(past_path)
    nx = r_init.shape[1]
    X_past = np.array(past_data.iloc[:, :nx])
    obj_past = np.array(past_data.iloc[:, nx]).reshape(-1,1)
    BO = ModularBayesianOptimization(model, domain_space, objective, acquisition, bo_evaluator, X_past, Y_init=obj_past)

max_time = None
max_iter = 2   # 100, 200, 500
tolerance = -np.inf

# Run Bayesian optimization
BO.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False)
BO.plot_convergence(os.path.join(save_dir, f'{file_tag}_convergence.png'))
# plt.savefig(f'res_{file_tag}_convergence.png')
evaluator.save_best_result()
#%%
optimal_rxn_coef = BO.x_opt
print("==============Optimal Result==============")
print("Rxn Coefficients for the First Reactor: ")
print(optimal_rxn_coef)
np.savetxt(os.path.join(save_dir, f'{file_tag}_optimal_coeff.txt'), optimal_rxn_coef)

optimal_result = hdo_prod_mse(optimal_rxn_coef.reshape(1, -1))
print(f"Mean Squared Error: {optimal_result}")

cc = sim.get_carbon_number_composition(sim.prod_stream)
with open(os.path.join(save_dir, f'{file_tag}_carbon_number_composition.txt'), 'w') as f:
    f.write(str(cc))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(cc.keys(), cc.values(), '-', label='Simulation')
plt.plot(sim.target.keys(), sim.target.values(), 'o', label='Experimental Data')
plt.legend()
plt.savefig(os.path.join(save_dir, f'{file_tag}_composition.png'))
plt.show()
#%%
sim.terminate()
