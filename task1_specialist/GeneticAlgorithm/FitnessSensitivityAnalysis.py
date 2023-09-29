# Import framework
import sys 
import glob, os
import argparse
from evoman.environment import Environment
from Controller import PlayerController
from GeneticOptimization import GeneticOptimization

# Import other libs
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_arguments():
    """
        Parses the input arguments.
        
        Args:
            plot (bool): A boolean flag to indicate whether to plot the results.
            experiment_name (str): The name of the experiment to be run.
            trials (int): The number of trials to run the simulation for.
    """
    parser = argparse.ArgumentParser(description="Your script description here")
    # Add input arguments
    parser.add_argument('-e', '--experiment_name', type=str, help='Directory to experiment')
    parser.add_argument('-t', '--trials', type=int, help='Integer value', default=3)
    return parser.parse_args()

def mkdir_experiment(experiment_name):
    """
        Sets the experiment name.
    """
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    else:
        os.system('rm -r ' + experiment_name)
        os.makedirs(experiment_name)

if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()

    # Set experiment name
    mkdir_experiment(args.experiment_name)
    enemy = re.search(r'enemy_(\d+)', args.experiment_name).group(1)
    print(enemy)

    experiment_time = time.time()  # Sets time marker

    # Define parameter space
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    
    # Initialize lists to store best fitness and best gain for each trial for each gamma and alpha pair
    config_cnt = 0
    for gamma, alpha in zip(gammas, alphas):
        print('---------------------------------------------------------------------------------')
        print('               GAMMA: ' + str(round(gamma, 1)) + ' ALPHA: ' + str(round(alpha, 1)))
        print('---------------------------------------------------------------------------------')

        # Add config folder
        config_name = os.path.join(args.experiment_name, 'gamma_' + str(round(gamma, 1)) + '_alpha_' + str(round(alpha, 1)))
        os.makedirs(config_name)

        mean_fitnesss = []

        # Run the simulation for the specified number of trials
        for t in range(args.trials):
            # Print experiment header
            print('---------------------------------------------------------------------------------')
            print('                               TRIAL: ' + str(t))
            print('---------------------------------------------------------------------------------')
            
            # Set trail name
            trail_name = os.path.join(config_name, 'trial_' + str(t))
            os.makedirs(trail_name)
            
            # Initialize game simulation in individual evolution mode, for single static enemy.
            env = Environment(experiment_name=trail_name,
                            enemies=[enemy],
                            playermode="ai",
                            player_controller=PlayerController(10),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
            # [NOTE]: Default environment fitness is assumed for experiment
            env.state_to_log() # Checks environment state

            trail_time = time.time()  # Sets time marker

            # Run Genetic Algorithm
            genetic = GeneticOptimization(env, trail_name, generations=12, gamma=gamma, alpha=alpha)
            best_solution, best_fitness = genetic.optimize()
            mean_fitnesss.append(np.mean(genetic.fitness_values))

            print('---------------------------------------------------------------------------------')
            print('TRIAL ' + str(t) + ' COMPLETED!')
            print('Total optimiztion time: ' + str(round(time.time() - trail_time)) + ' seconds')

        # Save data
        with open(os.path.join(args.experiment_name, 'SA_results.txt'), 'a') as file_aux:
            if config_cnt == 0: file_aux.write('gamma alpha avg_mean_fit')
            file_aux.write('\n' + str(round(gamma, 1)) + ' ' + str(round(alpha, 1)) + ' ' + str(round(np.mean(mean_fitnesss), 6)))

        config_cnt += 1

    print('---------------------------------------------------------------------------------')
    print('SENSITIVITY ANALYSIS COMPLETED!')
    print('Total experiment time: ' + str(round(time.time() - experiment_time)) + ' seconds')
