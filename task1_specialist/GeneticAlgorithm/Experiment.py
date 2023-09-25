# Import framework
import sys 
import glob, os
import argparse
from evoman.environment import Environment
from Controller import PlayerController
from GeneticOptimizationV2 import GeneticOptimization

# Import other libs
import time
import numpy as np
import re
from math import fabs,sqrt

def parse_arguments():
    """
        Parses the input arguments.
        
        Args:
            enemy (int): The enemy to be used to train the neural network.
            n_hidden_neurons (int): The number of hidden neurons in the neural network.
            trials (int): The number of trials to run the simulation for.
            visuals (bool): Whether to use visuals or not.
            mode (str): The mode to run the simulation in. Either 'train' or 'test'.
    """
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add input arguments
    parser.add_argument('-e', '--experiment_name', type=str, help='Directory to experiment')
    parser.add_argument('-n', '--n_hidden_neurons', type=int, help='Integer value', default=10)
    parser.add_argument('-t', '--trials', type=int, help='Integer value', default=10)
    parser.add_argument('-g', '--gens', type=int, help='Integer value', default=15)
    parser.add_argument('-v', '--visuals', type=bool, help='Boolean value', default=False)
    return parser.parse_args()

def check_visuals(visuals):
    """
        Checks whether to use visuals or not.
    """
    if not visuals:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

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
    enemy = re.search(r'enemy_(\d+)', args.experiment_name).group(1)


    # Check whether to turn on visuals
    check_visuals(args.visuals)

    # Set experiment name
    mkdir_experiment(args.experiment_name)

    experiment_time = time.time()  # Sets time marker

    # Run the simulation for the specified number of trials
    for t in range(args.trials):
        # Print experiment header
        print('---------------------------------------------------------------------------------')
        print('                               TRIAL: ' + str(t))
        print('---------------------------------------------------------------------------------')
        # Set trail name
        trail_name = args.experiment_name + '/trial_' + str(t)
        os.makedirs(trail_name)
        # Initialize game simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=trail_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=PlayerController(args.n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=args.visuals)
        # [NOTE]: Default environment fitness is assumed for experiment
        env.state_to_log() # Checks environment state
                
        trail_time = time.time()  # Sets time marker
        
        # Run Genetic Algorithm
        genetic = GeneticOptimization(env, trail_name)
        best_solution, best_fitness = genetic.optimize()

        print('---------------------------------------------------------------------------------')
        print('TRIAL ' + str(t) + ' COMPLETED!')
        print('Total optimiztion time: ' + str(round(time.time() - trail_time)) + ' seconds')
    
    print('---------------------------------------------------------------------------------')
    print('EXPERIMENT COMPLETED!')
    print('Total experiment time: ' + str(round(time.time() - experiment_time)) + ' seconds')