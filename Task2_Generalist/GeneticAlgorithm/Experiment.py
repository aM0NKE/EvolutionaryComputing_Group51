# Import framework
import sys 
import glob, os
import argparse
from evoman.environment import Environment
from Controller import GeneticController
from Optimization import GeneticOptimization

# Import other libs
import time
import re

def parse_arguments():
    """
        Parses the input arguments.
        
        Args:
            enemy (int): The enemy to be used to train the neural network.
            trials (int): The number of trials to run the simulation for.
    """
    parser = argparse.ArgumentParser(description="Your script description here")
    # Add input arguments
    parser.add_argument('-d', '--directory', type=str, help='String value', default='TESTING_GENETIC')
    parser.add_argument('-t', '--trials', type=int, help='Integer value', default=10)
    parser.add_argument('-g', '--gens', type=int, help='Integer value', default=25)
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

    # Make experiment directory
    mkdir_experiment(args.directory)

    experiment_time = time.time()  # Sets time marker

    # Run the simulation for the specified number of trials
    for t in range(args.trials):
        print('---------------------------------------------------------------------------------')
        print('                               TRIAL: ' + str(t))
        print('---------------------------------------------------------------------------------')
        # Set trail name
        trail_name = args.directory + '/trial_' + str(t)
        os.makedirs(trail_name)

        # Initialize game simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=trail_name,
                        enemies=[2,5,7,8],
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=GeneticController(10),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)
        # [NOTE]: Default environment fitness is assumed for experiment
        env.state_to_log() # Checks environment state
                
        trial_time = time.time()  # Sets time marker
        
        # Run Genetic Algorithm
        genetic = GeneticOptimization(env, trail_name, generations=args.gens)
        best_solution, best_fitness = genetic.optimize()

        print('---------------------------------------------------------------------------------')
        print('TRIAL ' + str(t) + ' COMPLETED!')
        print('Total optimiztion time: ' + str(round(time.time() - trial_time)) + ' seconds')
    
    print('---------------------------------------------------------------------------------')
    print('EXPERIMENT COMPLETED!')
    print('Total experiment time: ' + str(round(time.time() - experiment_time)) + ' seconds')