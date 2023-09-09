#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from Controller import PlayerController

# imports other libs
import numpy as np

enemy = 8 # choose the enemy (1 to 8)

experiment_name = 'genetic_v_enemy_'+str(enemy)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=PlayerController(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)


env.update_parameter('enemies',[enemy]) # update the enemy

# Load specialist controller
sol = np.loadtxt(experiment_name+'/best_solution.txt')
print('\n LOADING SAVED SPECIALIST SOLUTION FOR ENEMY '+str(enemy)+' \n')
env.play(sol)