# Import framework
import sys 
from evoman.environment import Environment
from Controller import PlayerController
from GeneticOptimizationV2 import GeneticOptimization

# Import other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import argparse

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

if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    # Check whether to turn on visuals
    check_visuals(args.visuals)

    # Set experiment name
    experiment_name = 'genetic_v_enemy_' + str(args.enemy)
    mkdir_experiment(experiment_name)

    # Initialize game simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[args.enemy],
                    playermode="ai",
                    player_controller=PlayerController(args.n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    # [NOTE]: Default environment fitness is assumed for experiment

    env.state_to_log() # Checks environment state
            
    ini = time.time()  # Sets time marker
    
    # Run Genetic Algorithm
    genetic = GeneticOptimization(env, args.mode, args.n_hidden_neurons, experiment_name)
    genetic.run()