# Import framework
import sys 
import glob, os
from evoman.environment import Environment
from Controller import PlayerController
from GeneticOptimization import GeneticOptimization

# Import other libs
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd

def mkdir_experiment(experiment_name):
    """
        Sets the experiment name.
    """
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    else:
        os.system('rm -r ' + experiment_name)
        os.makedirs(experiment_name)

# Make experiment directory
experiment_name = 'ParameterTuning'
mkdir_experiment(experiment_name)

# Define the range of values for each parameter
param_grid = {
    'selection_lambda': [10, 25, 50],
    'selection_k': [2, 4, 8],
    'mutation_rate': [0.01, 0.05, 0.1]
}
# Generate all possible combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

# Set default parameter values
n_hidden_neurons = 10 
dom_u = 1
dom_l = -1 
n_pop = 100
generations = 20
gamma = 0.9 
alpha = 0.1
crossover_alpha = 0.5

# Set other parameters
trials = 3
enemies = [5, 7, 8]

# Open the result file in append mode
result_file = os.path.join(experiment_name, 'results.txt')
with open(result_file, 'a') as f:
    # Write a header line
    f.write("Parameter Comb.\tEnemy\tTrial\tMean Fitness\n")

    # Loop through each parameter combination and run the genetic algorithm
    for params in tqdm(param_combinations):
        for enemy in enemies:
            for t in range(trials):
                # Initialize environment
                trial_name = f"EA_{str(params)}_enemy_{enemy}_trial_{t}"
                os.makedirs(os.path.join(experiment_name, trial_name))
                env = Environment(experiment_name=os.path.join(experiment_name, trial_name),
                                enemies=[enemy],
                                playermode="ai",
                                player_controller=PlayerController(n_hidden_neurons),
                                enemymode="static",
                                level=2,
                                speed="fastest",
                                visuals=False)

                # Run the genetic algorithm and store the result (fitness score)
                param_dict = dict(zip(param_grid.keys(), params))
                genetic_algorithm = GeneticOptimization(env, os.path.join(experiment_name, trial_name), n_hidden_neurons=n_hidden_neurons, dom_u=dom_u, dom_l=dom_l, n_pop=n_pop, generations=generations, gamma=gamma, alpha=alpha, crossover_alpha=crossover_alpha, selection_lambda=param_dict['selection_lambda'], selection_k=param_dict['selection_k'], mutation_rate=param_dict['mutation_rate'])
                best_sol, best_fitness = genetic_algorithm.optimize()

                # Store the mean fitness to .txt file
                mean_fitness = np.mean(genetic_algorithm.fitness_values)
                f.write(f"{str(params)}\t{enemy}\t{t}\t{mean_fitness}\n")

df = pd.read_csv('ParameterTuning/results.txt', sep='\t')
average_fitness = df.groupby('Parameter Comb.')['Mean Fitness'].mean().reset_index()
max = average_fitness.loc[average_fitness['Mean Fitness'].idxmax()]
print("Best parameter combination:")
print(max)