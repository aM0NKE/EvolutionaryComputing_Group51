# Import framework
import sys, os
from evoman.environment import Environment
from Controller import GeneticController

# Import other libs
import argparse
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
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
	parser.add_argument('-s', '--solution_name', type=str, help='Path to a solution.txt')
	parser.add_argument('-n', '--n_hidden_neurons', type=int, help='Integer value', default=10)
	return parser.parse_args()

if __name__ == "__main__":
    experiment_name = 'test'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Parse input arguments
    args = parse_arguments()

    enemies = [1,2,3,4,5,6,7,8]
    trials = 10

    with open(f'{args.solution_name[:-4]}_eval_results.txt', 'a') as f:
        f.write('enemy avg_player_hp avg_enemy_hp avg_gain avg_time win_percentage\n')

    total_gain = 0
    avg_gains = []
    for e in enemies:
        env = Environment(
                        experiment_name=experiment_name,
                        enemies=[e],
                        playermode="ai",
                        fullscreen=False,
                        player_controller=GeneticController(args.n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        sound="off",
                        speed="fastest")
        avg_playerHP = 0
        avg_enemyHP = 0 
        avg_gain = 0
        avg_time = 0
        wins = 0
        for t in range(trials):
            _, playerHP, enemyHP, time = env.play(pcont=np.loadtxt(args.solution_name))
            avg_playerHP += playerHP
            avg_enemyHP += enemyHP
            avg_gain += playerHP - enemyHP
            avg_time += time
            if playerHP > enemyHP:
                wins += 1
        
        avg_playerHP /= trials
        avg_enemyHP /= trials
        avg_gain /= trials
        avg_gains.append(avg_gain)
        total_gain += avg_gain
        avg_time /= trials
        win_perc = wins / trials
        with open(f'{args.solution_name[:-4]}_eval_results.txt', 'a') as f:
            f.write(f'{e} {round(avg_playerHP,2)} {round(avg_enemyHP,2)} {round(avg_gain,2)} {round(avg_time,2)} {round(win_perc,2)}\n')
    
    with open(f'{args.solution_name[:-4]}_eval_results.txt', 'a') as f:
        f.write(f'Total gain: {round(total_gain,2)}\n')
