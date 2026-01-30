import numpy as np
import pygad

# 1. LOAD EXPERIMENTAL DATA
# Replace this with your actual data loading (e.g., pd.read_csv)
# Example: Experimental data points for time t = [0, 1, 2, 3, 4]
exp_data_x = np.array([0, 1, 2, 3, 4])
exp_data_y = np.array([2.1, 2.9, 4.2, 5.8, 8.1])


# 2. DEFINE THE SIMULATION WRAPPER
def run_simulation(params):
    """
    This represents your simulation file.
    params: List of parameters to tune [a, b, c...]
    Returns: Simulated y values corresponding to exp_data_x
    """
    a, b = params  # Unpack parameters (genes)

    # --- SIMULATION LOGIC HERE ---
    # Example model: y = a * x + b
    # In reality, this might call an external script or complex function
    y_sim = a * exp_data_x + b
    # -----------------------------

    return y_sim


# 3. DEFINE FITNESS FUNCTION
# GA seeks to MAXIMIZE fitness. Since we want to MINIMIZE error,
# we define fitness as 1 / (error + epsilon) or -error.
def fitness_func(ga_instance, solution, solution_idx):
    # 'solution' contains the current parameters [a, b] proposed by GA

    # A. Run Simulation
    y_sim = run_simulation(solution)

    # B. Calculate Error (Sum of Squared Errors)
    error = np.sum((exp_data_y - y_sim) ** 2)

    # C. Convert to Fitness (Higher is better)
    # Adding a small epsilon (1e-8) prevents division by zero if error is 0
    fitness = 1.0 / (error + 1e-8)

    return fitness


# 4. CONFIGURE GENETIC ALGORITHM
ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=10,  # Population size (number of simulations per generation)
    num_genes=2,  # Number of parameters to tune (e.g., a and b)
    gene_type=float,
    init_range_low=0.0,  # Initial guess lower bound
    init_range_high=5.0,  # Initial guess upper bound
    mutation_percent_genes=10  # How much to mutate solutions
)

# 5. RUN OPTIMIZATION
print("Starting optimization...")
ga_instance.run()

# 6. RESULTS
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"--- Best Solution Found ---")
print(f"Parameters: {solution}")
print(f"Fitness: {solution_fitness}")

# Verify match
prediction = run_simulation(solution)
print(f"Experimental: {exp_data_y}")
print(f"Simulated:    {prediction}")