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

def training_plot():
	pass

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

    # Initialize results lists
    geneticA_gains = []
    geneticB_gains = []
    neatA_gains = []
    neatB_gains = []

    # Loop over all solutions
    for i, sol in enumerate([geneticA_sol, geneticB_sol, neatA_sol, neatB_sol]):
        print(i % 2)

        # If solution is from NEAT
        if type(sol) == type(neatA_sol):
            # Check the enemy group
            if i % 2 == 0:
                env = Environment(experiment_name=neatA_sol_dir,
                        enemies=enemy_groupA,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=NEATController(),
                        speed="fastest",
                        enemymode="static",
                        level=2,
                        visuals=False)
                for i in range(runs):
                    fitness, player_hp, enemy_hp, time = env.play(pcont=neatA_sol)
                    neatA_gains.append(player_hp - enemy_hp)

            else:
                env = Environment(experiment_name=neatB_sol_dir,
                        enemies=enemy_groupB,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=NEATController(),
                        speed="fastest",
                        enemymode="static",
                        level=2,
                        visuals=False)
                for i in range(runs):
                    fitness, player_hp, enemy_hp, time = env.play(pcont=neatB_sol)
                    neatB_gains.append(player_hp - enemy_hp)
                
        # Else solution is from GA 
        else:
            # Check the enemy group

            if i % 2 == 0:
                env = Environment(experiment_name=geneticA_sol_dir,
                        enemies=enemy_groupA,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=GeneticController(10),
                        speed="fastest",
                        enemymode="static",
                        level=2,
                        visuals=False)
                for i in range(runs):
                    fitness, player_hp, enemy_hp, time = env.play(pcont=geneticA_sol)
                    geneticA_gains.append(player_hp - enemy_hp)
            else:
                env = Environment(experiment_name=geneticB_sol_dir,
                        enemies=enemy_groupB,
                        multiplemode="yes",
                        playermode="ai",
                        player_controller=GeneticController(10),
                        speed="fastest",
                        enemymode="static",
                        level=2,
                        visuals=False)
                for i in range(runs):
                    fitness, player_hp, enemy_hp, time = env.play(pcont=geneticB_sol)
                    geneticB_gains.append(player_hp - enemy_hp)

    # Create boxplot
    data = [geneticA_gains, geneticB_gains, neatA_gains, neatB_gains]
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(['GA A', 'GA B', 'NEAT A', 'NEAT B'])
    plt.show()
        
if __name__ == "__main__":

    # Set enemy groups
    enemy_groupA = [2,5,7,8]
    enemy_groupB = [2,5,7,8]

    # Set experiment directories
    geneticA_dir = 'GeneticAlgorithm/TESTING_GENETIC'
    geneticB_dir = 'GeneticAlgorithm/TESTING_GENETIC'
    neatA_dir = 'NEAT/TESTING_NEAT'
    neatB_dir = 'NEAT/TESTING_NEAT'

    # Plot training progress
    training_plot()

    # Plot gain boxplot
    runs = 5
    boxplot(runs, enemy_groupA, enemy_groupB, geneticA_dir, geneticB_dir, neatA_dir, neatB_dir)
    