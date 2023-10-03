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
    plt.figure(figsize=(6, 4))

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
    plt.title(r'Enemy {} - $\gamma={}$/ $\alpha={}$'.format(enemy, gamma, alpha), fontsize=13)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save: 
        # Save plot
        plt.savefig(experiment_name + '/fitness_plot.png')
    else:
        plt.show()

def boxplot(save, experiment_name, compare_with, gains, gains2, enemy):
    """
        Args:
            save (bool): A boolean flag to indicate whether to save the plot.
            experiment_name (str): The name of the experiment to plot the results for.
            compare_with (str): The name of the experiment to compare with.
            gains (list): A list of gains for experiment_name.
            gains2 (list): A list of gains for compare_with.
            enemy (int): The enemy to plot the results for.
    """

    # do a t-test between gains and gains2
    t_statistic, p_value = stats.ttest_ind(gains, gains2)

    # Get gamma and alpha values from experiment name
    gamma = re.search(r'gamma_(\d+.\d+)', experiment_name).group(1)
    alpha = re.search(r'alpha_(\d+.\d+)', experiment_name).group(1)
    gamma2 = re.search(r'gamma_(\d+.\d+)', compare_with).group(1)
    alpha2 = re.search(r'alpha_(\d+.\d+)', compare_with).group(1)

    # Define custom colors for the boxplots
    colors = ['#4272f5', '#f5425a']

    # Define figure
    plt.figure(figsize=(4, 6))

    # Create a boxplot with custom colors
    boxplot = plt.boxplot([gains, gains2], labels=['The Striker', 'The Ninja'], patch_artist=True)

    # Set box colors
    for box, color in zip(boxplot['boxes'], colors):
        box.set(facecolor=color)

    # Set whisker and cap color
    for whisker, cap in zip(boxplot['whiskers'], boxplot['caps']):
        whisker.set(color='gray', linewidth=1)
        cap.set(color='gray', linewidth=1)

    # Set median line color
    for median in boxplot['medians']:
        median.set(color='black', linewidth=2)

    # Set the plot title with the t-test result
    plt.title(f'Enemy {enemy}\n(T-Test: t-stat={t_statistic:.2f}; p-value={p_value:.2f})', fontsize=14)
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Gain', fontsize=12)
    plt.tight_layout()

    if save: 
        plt.savefig(experiment_name + f'/box_{enemy}.png')
    else:
        # Save plot
        plt.show()
    
def eval_behavior(experiment_name, enemy, n_hidden_neurons=10, runs=5):

    # Get gamma and alpha values from experiment name
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
            try: accuracy = (100 - enemy_hp) / env.player_controller.shot_cnt
            except: accuracy = 0
            accs.append(round(accuracy, 2))

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
    df = pd.DataFrame({'enemy': [enemy for i in range(len(phps))], 'player_health': phps, 'enemy_health': ehps, 'gain': gains, 'time': times, 'lefts': lefts, 'rights': rights, 'jumps': jumps, 'releases': releases, 'shots': shots, 'accuracy': accs})
    df.to_csv(experiment_name + '/behavioral_evaluation.csv', index=False)

    return phps, ehps, gains, times, lefts, rights, jumps, releases, shots, accs

if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()
    enemy = re.search(r'enemy_(\d+)', args.experiment_name).group(1)
    compare_with = 'gamma_0.1_alpha_0.9_enemy_5'

    # Get mean, std, max fitness over all trials
    all_gens, all_mean_fit, all_std_fit, all_max_fit = [], [], [], []
    trials = glob.glob(args.experiment_name + '/trial*')
    for t in trials:
        data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
        all_gens.append(data['gen'])
        all_mean_fit.append(data['mean_fit'])
        all_std_fit.append(data['std_fit'])
        all_max_fit.append(data['max_fit'])

    # Eval behavior
    phps, ehps, gains, times, lefts, rights, jumps, releases, shots, accs = eval_behavior(args.experiment_name, enemy)
    phps2, ehps2, gains2, times2, lefts2, rights2, jumps2, releases2, shots2, accs2 = eval_behavior(compare_with, enemy)

    # Compare approaches
    boxplot(args.save, args.experiment_name, compare_with, gains, gains2, enemy)
    plot_fitness(args.save, args.experiment_name, enemy, all_gens, all_mean_fit, all_std_fit, all_max_fit)