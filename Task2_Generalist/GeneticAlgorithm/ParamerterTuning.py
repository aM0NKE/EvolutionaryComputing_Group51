# Import framework
import sys 
import glob, os
from evoman.environment import Environment
from Controller import GeneticController
from Optimization import GeneticOptimization

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
experiment_name = 'ParameterTuningV2'
mkdir_experiment(experiment_name)

# Define the range of values for each parameter
param_grid = {
    'n_pop': [50, 75, 100],
    'selection_lambda': [15, 25, 50], 
    'selection_k': [4, 8, 12], 
}
# Generate all possible combinations of parameters
param_combinations = list(itertools.product(*param_grid.values()))

# Set default parameter values
n_hidden_neurons=10
dom_u=1
dom_l=-1
generations=20
gamma=0.6
alpha=0.4
crossover_alpha=0.5

# Set other parameters
trials = 3
enemies = [1, 2, 3, 4, 5, 6, 7, 8]

# Open the result file in append mode
result_file = os.path.join(experiment_name, 'results.txt')
with open(result_file, 'a') as f:
    # Write a header line
    f.write("Parameter Comb. | Trial | Mean Fitness | Max Fitness | Mean Gain | Max Gain\n")

    # Loop through each parameter combination and run the genetic algorithm
    for params in tqdm(param_combinations):
        for t in range(trials):
            # Initialize environment
            trial_name = f"GA_{str(params)}_trial_{t}"
            os.makedirs(os.path.join(experiment_name, trial_name))
            # Initialize game simulation in individual evolution mode, for single static enemy.
            env = Environment(experiment_name=os.path.join(experiment_name, trial_name),
                            enemies=enemies,
                            multiplemode="yes",
                            playermode="ai",
                            player_controller=GeneticController(n_hidden_neurons),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)

            # Run the genetic algorithm and store the result (fitness score)
            param_dict = dict(zip(param_grid.keys(), params))
            genetic_algorithm = GeneticOptimization(env, os.path.join(experiment_name, trial_name), n_hidden_neurons=n_hidden_neurons, dom_u=dom_u, dom_l=dom_l, n_pop=param_dict['n_pop'], generations=generations, gamma=gamma, alpha=alpha, crossover_alpha=crossover_alpha, selection_lambda=param_dict['selection_lambda'], selection_k=param_dict['selection_k'])
            best_sol, best_fitness = genetic_algorithm.optimize()

            # Store the mean fitness to .txt file
            mean_fitness = np.mean(genetic_algorithm.fitness_values)
            max_fitness = np.max(genetic_algorithm.fitness_values)
            mean_gain = np.mean(genetic_algorithm.gain_values)
            max_gain = np.max(genetic_algorithm.gain_values)
            f.write(f"{str(params)}\t{t}\t{mean_fitness}\t{max_fitness}\t{mean_gain}\t{max_gain}\n")