# Import framework
import sys, os
from evoman.environment import Environment
from Controller import PlayerController

# Import other libs
import argparse
import glob
import re
import numpy as np
import pandas as pd

def parse_arguments():
	"""
        Parses the input arguments.
        
        Args:
            enemy (int): The enemy to be used to train the neural network.
            n_hidden_neurons (int): The number of hidden neurons in the neural network.
	"""
	parser = argparse.ArgumentParser(description="Your script description here")

	# Add input arguments
	parser.add_argument('-e', '--experiment_name', type=str, help='Directory to experiment')
	parser.add_argument('-n', '--n_hidden_neurons', type=int, help='Integer value', default=10)

	return parser.parse_args()

def find_best_solution(experiment_name):
	"""
        Finds the best solution for a given experiment.

        Args:
            experiment_name (str): The name of the experiment to find the best solution for.
    """
    # Initialize variables
	best_solution_trial_name = ''
	best_solution_gain = -100

	trials = glob.glob(experiment_name + '/trial_*')
	for t in trials:
		# Find the best solution for each trial
		data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
		current_solution = max(data['max_fit'])

		# Check if current solution is better than the best solution
		if current_solution > best_solution_gain:
			best_solution_gain = current_solution
			best_solution_trial_name = t

	return best_solution_trial_name

if __name__ == "__main__":

	# Parse input arguments
	args = parse_arguments()
	enemy = re.search(r'enemy_(\d+)', args.experiment_name).group(1)

	# Find best solution
	best_solution_trial_name = find_best_solution(args.experiment_name)

	# Initialize environment for single objective mode (specialist)  with static enemy and ai player
	env = Environment(experiment_name=best_solution_trial_name,
					enemies=[enemy],
					playermode="ai",
					player_controller=PlayerController(args.n_hidden_neurons),
					speed="normal",
					enemymode="static",
					level=2,
					visuals=True)

	# Load specialist controller
	sol = np.loadtxt(best_solution_trial_name + '/best_solution.txt')
	print('\nLOADING SAVED SPECIALIST SOLUTION FOR ENEMY ' + str(enemy) + ' \n')
	env.play(sol)