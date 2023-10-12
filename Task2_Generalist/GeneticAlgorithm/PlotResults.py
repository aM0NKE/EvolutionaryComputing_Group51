# Import framework
from evoman.environment import Environment
from Controller import GeneticController

# Import libs
import argparse
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def find_best_solution(experiment_name):
	"""
        Finds the best solution for a given experiment.

        Args:
            experiment_name (str): The name of the experiment to find the best solution for.
    """
    # Initialize variables
	best_solution_trial_name = ''
	best_solution_gain = -100

	trials = glob.glob(experiment_name + '/trial_*')
	for t in trials:
		# Find the best solution for each trial
		data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
		current_solution = max(data['max_fit'])

		# Check if current solution is better than the best solution
		if current_solution > best_solution_gain:
			best_solution_gain = current_solution
			best_solution_trial_name = t

	return best_solution_trial_name

def training_plot(enemy_group1, enemy_group2, GA1_EG1_dir, GA1_EG2_dir, GA2_EG1_dir, GA2_EG2_dir):
	
    # Loop over all directories
    for dir_n, dir in enumerate([GA1_EG1_dir, GA1_EG2_dir, GA2_EG1_dir, GA2_EG2_dir]):
        # Initialize lists for plotting
        all_gens, all_mean_fit, all_max_fit = [], [], []
        # Get all trials
        trials = glob.glob(dir + '/trial_*')
        # Loop over all trials
        for t in trials:
            # Read the optimization logs
            data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
            # Append data to lists
            all_gens.append(list(data.index.values))
            all_mean_fit.append(data['mean_fit'])
            all_max_fit.append(data['max_fit'])

        # Create list with mean over all trials
        gens = all_gens[0]
        # find highest fitness for each generation
        max_mean_fit = np.max(all_mean_fit, axis=0)
        min_mean_fit = np.min(all_mean_fit, axis=0)
        avg_mean_fit = np.mean(all_mean_fit, axis=0)
        max_max_fit = np.max(all_max_fit, axis=0)
        min_max_fit = np.min(all_max_fit, axis=0)
        avg_max_fit = np.mean(all_max_fit, axis=0)

        # Plot
        plt.figure(figsize=(9, 5))

        plt.plot(gens, avg_mean_fit, marker='o', linestyle='-', label='Avg. Mean Fitness', color='blue')
        plt.fill_between(gens, min_mean_fit, max_mean_fit, alpha=0.2, color='blue')
        plt.plot(gens, avg_max_fit, marker='o', linestyle='-', label='Avg. Max Fitness', color='red')
        plt.fill_between(gens, min_max_fit, max_max_fit, alpha=0.2, color='red')
        plt.title(f'GA: {["Fully-Random", "Fully Random", "Semi-Random", "Semi-Random"][dir_n]} | Enemies: {[enemy_group1, enemy_group2, enemy_group1, enemy_group2][dir_n]}', fontsize=22)
        plt.ylabel('Fitness', fontsize=20)
        plt.xlabel('Generation', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        plt.legend(loc='lower right', fontsize=18)
        plt.tight_layout()
        # plt.show()

        plt.savefig(f'{dir}/training_plot.png')

def boxplot(runs, enemy_group1, enemy_group2, GA1_EG1_dir, GA1_EG2_dir, GA2_EG1_dir, GA2_EG2_dir):

    enemies = [1,2,3,4,5,6,7,8]

    # Initialize dict for gains data
    gains = {GA1_EG1_dir: [], GA2_EG1_dir: [], GA1_EG2_dir: [], GA2_EG2_dir: []}

    for dir in [GA1_EG1_dir, GA1_EG2_dir, GA2_EG1_dir, GA2_EG2_dir]:
        trials = glob.glob(dir + '/trial*')
        for t in trials:
            total_gain = 0
            for e in enemies:
                # Initialize environment
                env = Environment(
                            experiment_name=t,
                            enemies=[e],
                            playermode="ai",
                            fullscreen=False,
                            player_controller=GeneticController(10),
                            enemymode="static",
                            level=2,
                            sound="off",
                            speed="fastest")
                            
                # Load best solution
                best_solution = np.loadtxt(t + '/best_solution.txt')
        
                # Play game runs times
                for i in range(runs):
                    # Play game
                    fitness, player_hp, enemy_hp, time = env.play(pcont=best_solution)
                    total_gain += player_hp - enemy_hp
            gains[dir].append(total_gain / runs)


    # T-test between total gains of both enemy groups
    t_statisticEG1, p_valueEG1 = stats.ttest_ind(gains[GA1_EG1_dir], gains[GA1_EG2_dir])
    t_statisticEG2, p_valueEG2 = stats.ttest_ind(gains[GA2_EG1_dir], gains[GA2_EG2_dir])

    colors = ['#4272f5', '#f5425a', '#4272f5', '#f5425a']
    plt.figure(figsize=(13, 7))
    boxplot = plt.boxplot(gains.values(), labels=['Fully-Rand. (All)', 'Semi-Rand. (All)', 'Fully-Rand. (1234)', 'Semi-Rand. (1234)'], patch_artist=True)
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
    plt.title(f'All: (t-stat={t_statisticEG1:.2f}; p-value={p_valueEG1:.2f}) | {enemy_group2}: (t-stat={t_statisticEG2:.2f}; p-value={p_valueEG2:.2f})', fontsize=22)
    plt.xlabel('Configuration', fontsize=20)
    plt.ylabel('Gain', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig('Gain.png')

        
if __name__ == "__main__":

    # Set enemy groups
    enemy_group1 = [1,2,3,4,5,7,8]
    enemy_group2 = [1,2,3,4]

    # Set experiment directories
    GA1_EG1_dir = 'GA1_EGALL_v2'
    GA1_EG2_dir = 'GA1_EG1234'
    GA2_EG1_dir = 'GA2_EGALL'
    GA2_EG2_dir = 'GA2_EG1234'

    # Plot training progress
    # training_plot(enemy_group1, enemy_group2, GA1_EG1_dir, GA1_EG2_dir, GA2_EG1_dir, GA2_EG2_dir)

    # Plot gain boxplot
    runs = 5
    boxplot(runs, enemy_group1, enemy_group2, GA1_EG1_dir, GA1_EG2_dir, GA2_EG1_dir, GA2_EG2_dir)
    