# Import framework
from evoman.environment import Environment
from NEATController import NEATPlayerController

# Import libs
import argparse
import glob
import pandas as pd
import pickle
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
    parser.add_argument('-s', '--show', type=bool, help='Show plots', default=False)
    return parser.parse_args()

def plot_fitness(show, experiment_name, data, all_mean_fit, all_std_fit, all_max_fit):
    # Define figure
    plt.figure()

    # Get data for plotting
    avg_mean_fit = [sum(x)/len(x) for x in zip(*all_mean_fit)]
    avg_std_fit = [sum(x)/len(x) for x in zip(*all_std_fit)]
    avg_max_fit = [sum(x)/len(x) for x in zip(*all_max_fit)]
    min_mean_fit = [min(x) for x in zip(*all_mean_fit)]
    max_mean_fit = [max(x) for x in zip(*all_mean_fit)]
    min_max_fit = [min(x) for x in zip(*all_max_fit)]
    max_max_fit = [max(x) for x in zip(*all_max_fit)]
    
    # Draw lines
    plt.plot(data.index, avg_mean_fit, marker='o', linestyle='-', label='Avg. Mean Fitness', color='blue')
    plt.fill_between(data.index, min_mean_fit, max_mean_fit, alpha=0.2, color='blue')
    plt.plot(data.index, avg_max_fit, marker='o', linestyle='-', label='Avg. Max Fitness', color='red')
    plt.fill_between(data.index, min_max_fit, max_max_fit, alpha=0.2, color='red')

    # Show plot
    plt.title('NEAT v. Enemy ' + experiment_name.split('_')[-1][0])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()

    if show: plt.show()
    else:
        # Save plot
        plt.savefig(experiment_name + '/fitness_plot.png')

def plot_gain(show, experiment_name, data, all_mean_gain, all_std_gain, all_max_gain):
    # Define figure
    plt.figure()
    
    # Get data for plotting
    avg_mean_gain = [sum(x)/len(x) for x in zip(*all_mean_gain)]
    avg_std_gain = [sum(x)/len(x) for x in zip(*all_std_gain)]
    avg_max_gain = [sum(x)/len(x) for x in zip(*all_max_gain)]
    min_mean_gain = [min(x) for x in zip(*all_mean_gain)]
    max_mean_gain = [max(x) for x in zip(*all_mean_gain)]
    min_max_gain = [min(x) for x in zip(*all_max_gain)]
    max_max_gain = [max(x) for x in zip(*all_max_gain)]
    
    # Draw lines
    plt.plot(data.index, avg_mean_gain, marker='o', linestyle='-', label='Avg. Mean Gain', color='blue')
    plt.fill_between(data.index, min_mean_gain, max_mean_gain, alpha=0.2, color='blue')
    plt.plot(data.index, avg_max_gain, marker='o', linestyle='-', label='Avg. Max Gain', color='red')
    plt.fill_between(data.index, min_max_gain, max_max_gain, alpha=0.2, color='red')

    # Show plot
    plt.title('NEAT v. Enemy ' + experiment_name.split('_')[-1][0])
    plt.xlabel('Generation')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend()

    if show: plt.show()
    else:
        # Save plot
        plt.savefig(experiment_name + '/gain_plot.png')
    
def boxplot(show, experiment_name, runs, enemy, n_hidden_neurons):
    # Define figure
    plt.figure()

    gains = []
    trials = glob.glob(experiment_name + '/trial*')
    for t in trials:
        for _ in range(runs):

            # Initialize environment
            env = Environment(experiment_name=t,
                            enemies=[enemy],
                            playermode="ai",
                            player_controller=NEATPlayerController(),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
            
            # Load best solution
            with open(t+'/best_solution', mode='rb') as file:
                best_solution = pickle.load(file)

            # Play game
            fitness, player_hp, enemy_hp, time = env.play(pcont=best_solution)

            # Calculate gain
            gain = player_hp - enemy_hp 
            gains.append(gain)
            
    # Draw boxplot
    plt.boxplot(gains, labels=['NEAT'])
    plt.xlabel('Approach')
    plt.ylabel('Gain')
    plt.title('NEAT v. Enemy ' + experiment_name.split('_')[-1][0])

    if show: plt.show()
    else:
        # Save plot
        plt.savefig(experiment_name + '/boxplot.png')
    

if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()
    enemy = args.experiment_name.split('_')[-1][0]

    all_mean_fit = []
    all_std_fit = []
    all_max_fit = []
    all_mean_gain = []
    all_std_gain = []
    all_max_gain = []

    # For each trial
    trials = glob.glob(args.experiment_name + '/trial*')
    for t in trials:
        # Get mean, std, max fitness over all trials
        data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
        all_mean_fit.append(data['mean_fit'])
        all_std_fit.append(data['std_fit'])
        all_max_fit.append(data['max_fit'])
        all_mean_gain.append(data['mean_gain'])
        all_std_gain.append(data['std_gain'])
        all_max_gain.append(data['max_gain'])

    boxplot(args.show, args.experiment_name, 10, enemy, 10)
    plot_fitness(args.show, args.experiment_name, data, all_mean_fit, all_std_fit, all_max_fit)
    plot_gain(args.show, args.experiment_name, data, all_mean_gain, all_std_gain, all_max_gain)

    # Use this function to perform a t-test between two approaches
    # stats.ttest_ind(a, b)