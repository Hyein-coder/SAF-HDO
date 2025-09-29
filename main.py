#%%
import os
from aspen_utils import AspenSim
from utils import CustomEvaluator
import GPy, GPyOpt
from GPyOpt.methods import ModularBayesianOptimization
import numpy as np


aspen_path = "D:/saf_hdo/aspen/250926_pyrolysis_oil_CC.apw"
sim = AspenSim(aspen_path)

cc = sim.get_carbon_number_composition(sim.prod_stream)
print(cc)

domain = [
    {'name': f'x{n_rxn}{i}', 'type': 'continuous', 'domain': (0, 1)} for n_rxn in sim.n_rxn for i in range(n_rxn)
]
r_now = sim.get_rxn_coefficients()

def hdo_prod_mse(x):
    sim.apply_rxn_coefficients(x)
    res = sim.get_carbon_number_composition(sim.prod_stream)
    mse = sum((sim.target[n] - res[n])**2 for n in sim.target.keys()) * 10
    print(mse)
    return mse
hdo_prod_mse(r_now)

#%%
evaluator = CustomEvaluator(hdo_prod_mse)
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
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# Set Bayesian optimization
BO = ModularBayesianOptimization(model, domain_space, objective, acquisition, evaluator, r_init)

max_time = None
max_iter = 5
tolerance = -np.inf

# Run Bayesian optimization
BO.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False)
BO.plot_convergence()

#%%
sim.terminate()
