# Import framework
import sys, os
from evoman.environment import Environment
from Controller import PlayerController

# Import other libs
import numpy as np
import argparse
import glob
import pandas as pd

def parse_arguments():
	"""
        Parses the input arguments.
        
        Args:
            enemy (int): The enemy to be used to train the neural network.
            n_hidden_neurons (int): The number of hidden neurons in the neural network.
            visuals (bool): Whether to use visuals or not.
            mode (str): The mode to run the simulation in. Either 'train' or 'test'.

	"""
	parser = argparse.ArgumentParser(description="Your script description here")

	# Add input arguments
	parser.add_argument('-e', '--enemy', type=int, help='Integer value between 1 and 8', default=8)
	parser.add_argument('-n', '--n_hidden_neurons', type=int, help='Integer value', default=10)
	parser.add_argument('-v', '--visuals', type=bool, help='Boolean value', default=False)
	parser.add_argument('-m', '--mode', type=str, help='String value (train or test)', default='train')
	parser.add_argument('-t', '--trials', type=int, help='Integer value', default=0)

	return parser.parse_args()

def find_best_solution(experiment_name):

	best_solution_trial_name = ''
	best_solution_fitness = 0

	trials = glob.glob(experiment_name + '/trial_*')
	for t in trials:
		# Find the best solution for each trial
		data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
		current_solution = data['max_gain'].iloc[-1]
		# Check if current solution is better than the best solution
		if current_solution > best_solution_fitness:
			best_solution_fitness = current_solution
			best_solution_trial_name = t

	return best_solution_trial_name

if __name__ == "__main__":

	# Parse input arguments
	args = parse_arguments()
	
	# Set experiment name
	experiment_name = 'genetic_v_enemy_' + str(args.enemy)

	# Find best solution
	best_solution_trial_name = find_best_solution(experiment_name)

	# Initialize environment for single objective mode (specialist)  with static enemy and ai player
	env = Environment(experiment_name=best_solution_trial_name,
					playermode="ai",
					player_controller=PlayerController(args.n_hidden_neurons),
					speed="normal",
					enemymode="static",
					level=2,
					visuals=True)
	env.update_parameter('enemies',[args.enemy]) # update the enemy

	# Load specialist controller
	sol = np.loadtxt(best_solution_trial_name+'/best_solution.txt')
	print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(args.enemy)+' \n')
	env.play(sol)