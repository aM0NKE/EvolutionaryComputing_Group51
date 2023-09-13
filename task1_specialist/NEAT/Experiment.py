# Import framework
import sys 
from evoman.environment import Environment
from NEATController import NEATPlayerController
from NEATOptimization import NEATOptimization

# Import other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import argparse
import neat

def parse_arguments():
    """
        Parses the input arguments.
        
        Args:
            enemy (int): The enemy to be used to train the neural network.
            trials (int): The number of trials to run the simulation for.
            visuals (bool): Whether to use visuals or not.
    """
    parser = argparse.ArgumentParser(description="Your script description here")
    # Add input arguments
    parser.add_argument('-e', '--enemy', type=int, help='Integer value between 1 and 8', default=8)
    parser.add_argument('-t', '--trials', type=int, help='Integer value', default=10)
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

    # Check whether to turn on visuals
    check_visuals(args.visuals)

    # Set experiment name
    experiment_name = 'neat_v_enemy_' + str(args.enemy)
    mkdir_experiment(experiment_name)

    experiment_time = time.time()  # Sets time marker for entire experiment

    # Run the simulation for the specified number of trials
    for t in range(args.trials):
        print('---------------------------------------------------------------------------------')
        print('                               TRIAL: ' + str(t))
        print('---------------------------------------------------------------------------------')
        
        # Set trail name
        trail_name = experiment_name + '/trial_' + str(t)
        os.makedirs(trail_name)
        
        # Initialize game simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=trail_name,
                        enemies=[args.enemy],
                        playermode="ai",
                        player_controller=NEATPlayerController(),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=args.visuals)
        # [NOTE]: Default environment fitness is assumed for experiment
        env.state_to_log() # Checks environment state
                
        trail_time = time.time()  # Sets time marker for individual trial
        
        # Set NEAT parameters
        gens = 25
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'NEATConfig')

        # Run NEAT optimization
        neat_obj = NEATOptimization(env, trail_name, gens, neat_config)
        neat_obj._run()

        print('---------------------------------------------------------------------------------')
        print('TRIAL ' + str(t) + ' COMPLETED!')
        print('Total optimiztion time: ' + str(round(time.time() - trail_time)) + ' seconds')
    
    print('---------------------------------------------------------------------------------')
    print('EXPERIMENT COMPLETED!')
    print('Total experiment time: ' + str(round(time.time() - experiment_time)) + ' seconds')