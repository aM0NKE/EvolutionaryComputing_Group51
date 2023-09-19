# Import framework
import sys 
from evoman.environment import Environment
from Controller import PlayerController
from GeneticOptimizationV2 import GeneticOptimization

# Import other libs
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import fabs,sqrt
import glob, os
import argparse

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
    parser.add_argument('-e', '--enemy', type=int, help='Integer value between 1 and 8', default=1)
    parser.add_argument('-n', '--n_hidden_neurons', type=int, help='Integer value', default=10)
    parser.add_argument('-p', '--plot', type=bool, help='Boolean value', default=True)
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

def plot_SA_results(enemy):
    df = pd.read_csv('SA_enemy_1/SA_results.txt', sep=" ")

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points
    ax.scatter(df['gamma'], df['alpha'], df['mean_best_fit'], c=df['mean_best_gain'], cmap='viridis')

    # Set labels for the axes
    ax.set_title('Sensitivity Analysis (Enemy ' + str(enemy) + '): Fitness Function')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('Mean Best Fitness')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()

    if args.plot:
        plot_SA_results(args.enemy)

    else:

        # Check whether to turn on visuals
        check_visuals(args.visuals)

        # Set experiment name
        experiment_name = 'SA_enemy_' + str(args.enemy)
        mkdir_experiment(experiment_name)

        experiment_time = time.time()  # Sets time marker

        # Define parameter space
        gammas = np.linspace(0.1, 0.9, 9)
        alphas = np.linspace(0.9, 0.1, 9)
        
        # Initialize lists to store best fitness and best gain for each trial for each gamma and alpha pair
        best_fitnesss, best_gains = [], []
        for i, (gamma, alpha) in enumerate(zip(gammas, alphas)):
            print('---------------------------------------------------------------------------------')
            print('                               GAMMA: ' + str(gamma) + ' ALPHA: ' + str(alpha))
            print('---------------------------------------------------------------------------------')

            # Add config folder
            config_name = os.path.join(experiment_name, 'gamma_' + str(gamma) + '_alpha_' + str(alpha))
            os.makedirs(config_name)

            best_fitnesss.append([])
            best_gains.append([])

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
                                enemies=[args.enemy],
                                playermode="ai",
                                player_controller=PlayerController(args.n_hidden_neurons),
                                enemymode="static",
                                level=2,
                                speed="fastest",
                                visuals=args.visuals)
                # [NOTE]: Default environment fitness is assumed for experiment
                env.state_to_log() # Checks environment state

                trail_time = time.time()  # Sets time marker

                # Set Genetic Algorithm parameters
                n_hidden_neurons = args.n_hidden_neurons
                dom_u = 1
                dom_l = -1
                npop = 100
                gens = 20
                selection_prob = 0.2
                crossover_prob = 0.2
                mutation_prob = 0.2
                # Run Genetic Algorithm
                genetic = GeneticOptimization(env, trail_name, n_hidden_neurons, dom_u, dom_l, npop, gens, selection_prob, crossover_prob, mutation_prob, gamma, alpha)
                genetic.run()
                best_fitnesss[i].append(genetic.best_fit)
                best_gains[i].append(genetic.best_gain)

                print('---------------------------------------------------------------------------------')
                print('TRIAL ' + str(t) + ' COMPLETED!')
                print('Total optimiztion time: ' + str(round(time.time() - trail_time)) + ' seconds')
            
            with open(os.path.join(experiment_name, 'SA_results.txt'), 'a') as file_aux:
                if i == 0: file_aux.write('gamma alpha mean_best_fit mean_best_gain')
                file_aux.write('\n' + str(gamma) + ' ' + str(alpha) + ' ' + str(np.mean(best_fitnesss[i])) + ' ' + str(np.mean(best_gains[i])))

        print('---------------------------------------------------------------------------------')
        print('SENSITIVITY ANALYSIS COMPLETED!')
        print('Total experiment time: ' + str(round(time.time() - experiment_time)) + ' seconds')
