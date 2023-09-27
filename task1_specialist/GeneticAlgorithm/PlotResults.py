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
    
    # Get gamma and alpha values from experiment name (e.g. enemy_1/gamma_0.5/alpha_0.5)
    gamma = re.search(r'gamma_(\d+.\d+)', experiment_name).group(1)
    alpha = re.search(r'alpha_(\d+.\d+)', experiment_name).group(1)
    
    # Define figure
    plt.figure()

    # Get data for plotting
    min_mean_fit = [min(x) for x in zip(*all_mean_fit)]
    avg_mean_fit = [sum(x)/len(x) for x in zip(*all_mean_fit)]
    max_mean_fit = [max(x) for x in zip(*all_mean_fit)]

    min_max_fit = [min(x) for x in zip(*all_max_fit)]
    avg_max_fit = [sum(x)/len(x) for x in zip(*all_max_fit)]
    max_max_fit = [max(x) for x in zip(*all_max_fit)]
    
    # Draw lines
    plt.plot(all_gens[0], avg_mean_fit, marker='o', linestyle='-', label='Avg. Mean Fitness', color='blue')
    plt.fill_between(all_gens[0], min_mean_fit, max_mean_fit, alpha=0.2, color='blue')
    plt.plot(all_gens[0], avg_max_fit, marker='o', linestyle='-', label='Avg. Max Fitness', color='red')
    plt.fill_between(all_gens[0], min_max_fit, max_max_fit, alpha=0.2, color='red')

    # Show plot
    plt.title(f'gamma={gamma}/alpha={alpha} (enemy={enemy})')
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
     # Get gamma and alpha values from experiment name (e.g. enemy_1/gamma_0.5/alpha_0.5)
    gamma = re.search(r'gamma_(\d+.\d+)', experiment_name).group(1)
    alpha = re.search(r'alpha_(\d+.\d+)', experiment_name).group(1)

    # Get data for plotting
    phps, ehps, gains, times, = [], [], [], []
    lefts, rights, jumps, shots, accs, releases = [], [], [], [], [], []
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

        # Play game runs times
        for i in range(runs):
            # Play game
            fitness, player_hp, enemy_hp, time = env.play(pcont=best_solution)

            phps.append(round(player_hp, 2))
            ehps.append(round(enemy_hp, 2))
            # Calculate gain
            gain = player_hp - enemy_hp 
            gains.append(round(gain, 2))
            times.append(time)

            lefts.append(env.player_controller.lefts)
            rights.append(env.player_controller.rights)
            jumps.append(env.player_controller.jumps)
            shots.append(env.player_controller.shot_cnt)
            releases.append(env.player_controller.releases)
            # Calculate accuracy
            accuracy = (100 - enemy_hp) / env.player_controller.shot_cnt
            accs.append(round(accuracy, 2))

    # Define figure
    plt.figure()
            
    # Draw boxplot
    plt.boxplot(gains, labels=['Gain'])
    plt.xlabel('Approach')
    plt.ylabel('Gain')
    plt.title(f'gamma={gamma}/alpha={alpha} (enemy={enemy})')

    if save: 
        plt.savefig(experiment_name + '/boxplot.png')
    else:
        # Save plot
        plt.show()

    # Print statistics
    print("\nSTATISTICS:")
    print("Average Player HP:   ", np.mean(phps))
    print("Average Enemy HP:    ", np.mean(ehps))
    print("Average Gain:        ", np.mean(gains))
    print("Average Time:        ", np.mean(times))
    print("Average Lefts:       ", np.mean(lefts))
    print("Average Rights:      ", np.mean(rights))
    print("Average Jumps:       ", np.mean(jumps))
    print("Average Releases:    ", np.mean(releases))
    print("Average Total Shots: ", np.mean(shots))
    print("Average Damage/Shot: ", np.mean(accs))
    # Save statistics as dataframe
    df = pd.DataFrame({'player_health': phps, 'enemy_health': ehps, 'gain': gains, 'time': times, 'lefts': lefts, 'rights': rights, 'jumps': jumps, 'releases': releases, 'shots': shots, 'accuracy': accs})
    df.to_csv(experiment_name + '/behavioral_evaluation.csv', index=False)
    
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