# Import framework
import sys, os
from evoman.environment import Environment
from NEATController import NEATPlayerController

# Import other libs
import neat
import numpy as np
import pandas as pd
import pickle
import argparse
import glob

def parse_arguments():
	"""
        Parses the input arguments.
        
        Args:
            enemy (int): The enemy to be used to train the neural network.
            n_hidden_neurons (int): The number of hidden neurons in the neural network.
	"""
	parser = argparse.ArgumentParser(description="Your script description here")

	# Add input arguments
	parser.add_argument('-e', '--enemy', type=int, help='Integer value between 1 and 8', default=8)
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

    # Find all trials
    trials = glob.glob(experiment_name + '/trial_*')
    for t in trials:
        # Find the best solution for each trial
        data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
        current_solution = max(data['max_gain'])

        # Check if current solution is better than the best solution
        if current_solution > best_solution_gain:
            best_solution_gain = current_solution
            best_solution_trial_name = t

    return best_solution_trial_name

def play_game(env, sol):
    """
        Plays the game with a given solution.
        
        Args:
            env (Environment): The environment to be used.
            sol (list): The solution to be used.
    """
    # Play game
    fitness, player_hp, enemy_hp, time = env.play(pcont=sol)
	
    # Calculate gain
    gain = player_hp - enemy_hp
	
    return fitness, gain

if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()

    # Set experiment name
    experiment_name = 'neat_v_enemy_' + str(args.enemy)

    # Find best solution
    best_solution_trial_name = find_best_solution(experiment_name)

    # Initialize environment for single objective mode (specialist)  with static enemy and ai player
    env = Environment(experiment_name=best_solution_trial_name,
                    enemies=[args.enemy],
                    playermode="ai",
                    player_controller=NEATPlayerController(),
                    speed="normal",
                    enemymode="static",
                    level=2,
                    visuals=True)
	
    # Load best solution
    with open(best_solution_trial_name+'/best_solution', mode='rb') as file:
        sol = pickle.load(file)

    # Play game with best solution
    play_game(env, sol)