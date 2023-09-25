# Import framework
from evoman.environment import Environment
from Controller import PlayerController

# Import libs
import argparse
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def parse_arguments():
    """
        Parses the input arguments.
        
        Args:
            experiment_name (str): The name of the experiment to plot the results for.
    """
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add input arguments
    parser.add_argument('-e', '--experiment_name', type=str, help='Directory to experiment')
    parser.add_argument('-s', '--save', action='store_true', help='Save plots')
    return parser.parse_args()

def plot_fitness(save, experiment_name, enemy, all_gens, all_mean_fit, all_std_fit, all_max_fit):
    # Define figure
    plt.figure()

    # Get data for plotting
    min_mean_fit = [min(x) for x in zip(*all_mean_fit)]
    avg_mean_fit = [sum(x)/len(x) for x in zip(*all_mean_fit)]
    max_mean_fit = [max(x) for x in zip(*all_mean_fit)]

    avg_std_fit = [sum(x)/len(x) for x in zip(*all_std_fit)]

    min_max_fit = [min(x) for x in zip(*all_max_fit)]
    avg_max_fit = [sum(x)/len(x) for x in zip(*all_max_fit)]
    max_max_fit = [max(x) for x in zip(*all_max_fit)]
    
    # Draw lines
    plt.plot(all_gens[0], avg_mean_fit, marker='o', linestyle='-', label='Avg. Mean Fitness', color='blue')
    plt.fill_between(all_gens[0], min_mean_fit, max_mean_fit, alpha=0.2, color='blue')
    # plt.fill_between(all_gens[0], [x-y for x, y in zip(avg_mean_fit,avg_std_fit)], [x+y for x, y in zip(avg_mean_fit,avg_std_fit)], alpha=0.2, color='blue')
    plt.plot(all_gens[0], avg_max_fit, marker='o', linestyle='-', label='Avg. Max Fitness', color='red')
    plt.fill_between(all_gens[0], min_max_fit, max_max_fit, alpha=0.2, color='red')

    # Show plot
    plt.title('Genetic v. Enemy ' + str(enemy))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend(loc='lower right')

    if save: 
        # Save plot
        plt.savefig(experiment_name + '/fitness_plot.png')
    else:
        plt.show()

def boxplot(save, experiment_name, enemy, n_hidden_neurons, runs):
    # Define figure
    plt.figure()

    gains, accs = [], []
    trials = glob.glob(experiment_name + '/trial*')
    for t in trials:
        # Initialize environment
        env = Environment(experiment_name=t,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=PlayerController(n_hidden_neurons),
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)
                    
        # Load best solution
        best_solution = np.loadtxt(t + '/best_solution.txt')

        # Play game
        fitness, player_hp, enemy_hp, time = env.play(pcont=best_solution)

        # Calculate accuracy
        accuracy = (100-enemy_hp)/env.player_controller.shot_cnt
        accs.append(accuracy)

        # Calculate gain
        gain = player_hp - enemy_hp 
        gains.append(gain)
            
    # Draw boxplot
    plt.boxplot(gains, labels=['Gain'])
    plt.xlabel('Approach')
    plt.ylabel('Gain')
    plt.title('Genetic v. Enemy ' + str(enemy))

    if save: 
        plt.savefig(experiment_name + '/boxplot.png')
    else:
        # Save plot
        plt.show()

    print("Mean Damage/Shot: ", np.mean(accs))
    
if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()
    enemy = re.search(r'enemy_(\d+)', args.experiment_name).group(1)

    # Get mean, std, max fitness over all trials
    all_gens, all_mean_fit, all_std_fit, all_max_fit = [], [], [], []
    trials = glob.glob(args.experiment_name + '/trial*')
    for t in trials:
        data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
        all_gens.append(data['gen'])
        all_mean_fit.append(data['mean_fit'])
        all_std_fit.append(data['std_fit'])
        all_max_fit.append(data['max_fit'])

    # Plot results
    boxplot(args.save, args.experiment_name, enemy, 10, 5)
    plot_fitness(args.save, args.experiment_name, enemy, all_gens, all_mean_fit, all_std_fit, all_max_fit)

    # Use this function to perform a t-test between two approaches
    # stats.ttest_ind(a, b)