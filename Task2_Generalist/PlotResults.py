# # Import framework
from evoman.environment import Environment
from GeneticAlgorithm.Controller import GeneticController
from NEAT.Controller import NEATController

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

def training_plot(enemy_groupA, enemy_groupB, geneticA_dir, geneticB_dir, neatA_dir, neatB_dir):
	
    # Initialize lists for plotting
    all_gens, all_mean_fit, all_std_fit, all_max_fit = [], [], [], []

    # Loop over all directories
    for dir_n, dir in enumerate([geneticA_dir, geneticB_dir, neatA_dir, neatB_dir]):
        all_gens.append([]), all_mean_fit.append([]), all_std_fit.append([]), all_max_fit.append([])

        # Get all trials
        trials = glob.glob(dir + '/trial_*')
        # Loop over all trials
        for t in trials:
            # Read the optimization logs
            data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
            # Append data to lists
            all_gens[dir_n].append(data.index.values)
            all_mean_fit[dir_n].append(data['mean_fit'])
            all_std_fit[dir_n].append(data['std_fit'])
            all_max_fit[dir_n].append(data['max_fit'])

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(all_gens[0][0], all_mean_fit[0][0], label=f'GA {str(enemy_groupA)}')
    ax.fill_between(all_gens[0][0], all_mean_fit[0][0] - all_std_fit[0][0], all_mean_fit[0][0] + all_std_fit[0][0], alpha=0.2)
    ax.plot(all_gens[1][0], all_mean_fit[1][0], label=f'GA {str(enemy_groupB)}')
    ax.fill_between(all_gens[1][0], all_mean_fit[1][0] - all_std_fit[1][0], all_mean_fit[1][0] + all_std_fit[1][0], alpha=0.2)
    ax.plot(all_gens[2][0], all_mean_fit[2][0], label=f'NEAT {str(enemy_groupA)}')
    ax.fill_between(all_gens[2][0], all_mean_fit[2][0] - all_std_fit[2][0], all_mean_fit[2][0] + all_std_fit[2][0], alpha=0.2)
    ax.plot(all_gens[3][0], all_mean_fit[3][0], label=f'NEAT {str(enemy_groupB)}')
    ax.fill_between(all_gens[3][0], all_mean_fit[3][0] - all_std_fit[3][0], all_mean_fit[3][0] + all_std_fit[3][0], alpha=0.2)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.legend()
    plt.show()        

def boxplot(runs, enemy_groupA, enemy_groupB, geneticA_dir, geneticB_dir, neatA_dir, neatB_dir):

    # Find best Genetic solutions for both enemy groups
    geneticA_sol_dir = find_best_solution(geneticA_dir)
    geneticB_sol_dir = find_best_solution(geneticB_dir)
    geneticA_sol = np.loadtxt(geneticA_sol_dir + '/best_solution.txt')
    geneticB_sol = np.loadtxt(geneticB_sol_dir + '/best_solution.txt')

    # Find best NEAT solutions for both enemy groups
    neatA_sol_dir = find_best_solution(neatA_dir)
    neatB_sol_dir = find_best_solution(neatB_dir)
    with open(neatA_sol_dir + '/best_solution', mode='rb') as file:
        neatA_sol = pickle.load(file)
    with open(neatB_sol_dir + '/best_solution', mode='rb') as file:
        neatB_sol = pickle.load(file)
    
    # Save solutions in dict
    solutions = {geneticA_sol_dir: geneticA_sol, geneticB_sol_dir: geneticB_sol, neatA_sol_dir: neatA_sol, neatB_sol_dir: neatB_sol}

    # Initialize dict for gains data
    gains = {geneticA_sol_dir: [], geneticB_sol_dir: [], neatA_sol_dir: [], neatB_sol_dir: []}

    # Loop over all solutions
    for dir, sol in solutions.items():
        # If solution is from NEAT
        if type(sol) == type(neatA_sol):
            env = Environment(experiment_name=dir,
                    enemies=[1,2,3,4,5,6,7,8],
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=NEATController(),
                    speed="fastest",
                    enemymode="static",
                    level=2,
                    visuals=False)
        # Else solution is from GA 
        else:
            env = Environment(experiment_name=dir,
                    enemies=[1,2,3,4,5,6,7,8],
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=GeneticController(10),
                    speed="fastest",
                    enemymode="static",
                    level=2,
                    visuals=False)
            
        # Run game multiple times
        for i in range(runs):
            fitness, player_hp, enemy_hp, time = env.play(pcont=sol)
            gains[dir].append(player_hp - enemy_hp)

    # Create boxplot
    fig, ax = plt.subplots()
    ax.boxplot(gains.values())
    ax.set_xticklabels([f'GA {str(enemy_groupA)}', f'GA {str(enemy_groupB)}', f'NEAT {str(enemy_groupA)}', f'NEAT {str(enemy_groupB)}'])
    plt.ylabel('Gain')
    plt.xlabel('Algorithm')
    plt.show()
        
if __name__ == "__main__":

    # Set enemy groups
    enemy_groupA = [2,5,7,8]
    enemy_groupB = [2,5,7,8]

    # Set experiment directories
    geneticA_dir = 'GeneticAlgorithm/TESTING_GENETIC'
    geneticB_dir = 'GeneticAlgorithm/TESTING2_GENETIC'
    neatA_dir = 'NEAT/TESTING_NEAT'
    neatB_dir = 'NEAT/TESTING2_NEAT'

    # Plot training progress
    training_plot(enemy_groupA, enemy_groupB, geneticA_dir, geneticB_dir, neatA_dir, neatB_dir)

    # Plot gain boxplot
    runs = 5
    boxplot(runs, enemy_groupA, enemy_groupB, geneticA_dir, geneticB_dir, neatA_dir, neatB_dir)
    