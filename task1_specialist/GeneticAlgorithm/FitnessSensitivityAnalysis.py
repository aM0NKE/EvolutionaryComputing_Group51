# Import framework
import sys 
from evoman.environment import Environment
from Controller import PlayerController
from GeneticOptimizationV2 import GeneticOptimization

# Import other libs
import argparse
import glob, os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    parser.add_argument('-p', '--plot', action='store_true', help='A boolean flag.')
    parser.add_argument('-e', '--experiment_name', type=str, help='Directory to experiment')
    parser.add_argument('-n', '--n_hidden_neurons', type=int, help='Integer value', default=10)
    parser.add_argument('-t', '--trials', type=int, help='Integer value', default=5)
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

def plot_SA_results(experiment_name):
    # Extract enemy
    enemy = re.search(r'enemy_(\d+)', experiment_name).group(1)

    # Read data
    df = pd.read_csv(os.path.join(experiment_name, 'SA_results.txt'), sep=" ")
    gamma = df['gamma'].values
    alpha = df['alpha'].values
    avg_mean_fit = df['avg_mean_fit'].values

    # Create a figure and a 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create positions for the diagonal bars
    positions = [(g, a) for g, a in zip(gamma, alpha)]

    # Color the bars by their `avg_mean_fit` values
    cmap = plt.get_cmap('viridis')  # You can choose a different colormap if you prefer
    norm = plt.Normalize(min(avg_mean_fit), max(avg_mean_fit))
    colors = cmap(norm(avg_mean_fit))

    # Create the 3D bars on the diagonal axes
    for i, (g, a) in enumerate(positions):
        ax.bar3d(g, a, 0, 0.1, 0.1, avg_mean_fit[i], shade=True, color=colors[i])

    # Set labels for the axes
    ax.set_title('SA - Fitness Function: Enemy ' + str(enemy))
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('Avg Mean Fit')

    # Show the plot
    plt.show()
    # plt.savefig(os.path.join(experiment_name, 'SA_results.png'))

if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()

    if args.plot:
        plot_SA_results(args.experiment_name)

    else:
        # Check whether to turn on visuals
        check_visuals(args.visuals)

        # Set experiment name
        mkdir_experiment(args.experiment_name)
        enemy = re.search(r'enemy_(\d+)', args.experiment_name).group(1)

        experiment_time = time.time()  # Sets time marker

        # Define parameter space
        # gammas = np.linspace(0.0, 1.0, 11)
        # alphas = np.linspace(0.0, 1.0, 11)
        gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        alphas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        # Initialize lists to store best fitness and best gain for each trial for each gamma and alpha pair
        mean_fitnesss, mean_gains = [], []
        cnt = 0
        # for gamma in gammas:
        #     for alpha in alphas:
        for gamma, alpha in zip(gammas, alphas):
            print('---------------------------------------------------------------------------------')
            print('                               GAMMA: ' + str(round(gamma, 1)) + ' ALPHA: ' + str(round(alpha, 1)))
            print('---------------------------------------------------------------------------------')

            # Add config folder
            config_name = os.path.join(args.experiment_name, 'gamma_' + str(round(gamma, 1)) + '_alpha_' + str(round(alpha, 1)))
            os.makedirs(config_name)

            mean_fitnesss.append([])
            mean_gains.append([])

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
                                player_controller=PlayerController(args.n_hidden_neurons),
                                enemymode="static",
                                level=2,
                                speed="fastest",
                                visuals=args.visuals)
                # [NOTE]: Default environment fitness is assumed for experiment
                env.state_to_log() # Checks environment state

                trail_time = time.time()  # Sets time marker

                # Run Genetic Algorithm
                genetic = GeneticOptimization(env, trail_name, gamma=gamma, alpha=alpha)
                best_solution, best_fitness = genetic.optimize()
                mean_fitnesss[cnt].append(np.mean(genetic.fitness_values))
                # mean_gains[cnt].append(np.mean(genetic.gain_values))

                print('---------------------------------------------------------------------------------')
                print('TRIAL ' + str(t) + ' COMPLETED!')
                print('Total optimiztion time: ' + str(round(time.time() - trail_time)) + ' seconds')

            with open(os.path.join(args.experiment_name, 'SA_results.txt'), 'a') as file_aux:
                if cnt == 0: file_aux.write('gamma alpha avg_mean_fit')
                file_aux.write('\n' + str(round(gamma, 1)) + ' ' + str(round(alpha, 1)) + ' ' + str(round(np.mean(mean_fitnesss[cnt]), 6)))

            cnt += 1

        print('---------------------------------------------------------------------------------')
        print('SENSITIVITY ANALYSIS COMPLETED!')
        print('Total experiment time: ' + str(round(time.time() - experiment_time)) + ' seconds')
