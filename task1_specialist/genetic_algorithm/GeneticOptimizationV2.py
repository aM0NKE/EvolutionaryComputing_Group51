#########################################################################################         
#                        This is our implementation of the:                             #
#                Genetic Algorithm for Optimizing a Neural Network.                     #
#           (The controller -that contains the NN- is demo_controller.py)               #
#########################################################################################

# Import framework
import sys 
from evoman.environment import Environment
from Controller import PlayerController

# Import other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import argparse


#########################################################################################
#                                [THE GENETIC ALGORITHM]                                #
#   Optimization for controller solution (best genotype-weights for phenotype-network)  #
#########################################################################################
class GeneticOptimization(object):
    """
    This class implements the Genetic Algorithm used to find the best strategies
    to beat the Evoman enemies.

    Args:
        env (Environment): The Evoman game environment to be used.
        mode (str): The mode to run the simulation in. Either 'train' or 'test'.
        n_hidden_neurons (int): The number of hidden neurons in the neural network.
        experiment_name (str): The name of the experiment to be run.
    """

    def __init__(self, env, mode, n_hidden_neurons, experiment_name):
        """
            Initializes the Genetic Algorithm.
            
            Args:
                env (Environment): The Evoman game environment to be used.
                mode (str): The mode to run the simulation in. Either 'train' or 'test'.
                n_hidden_neurons (int): The number of hidden neurons in the neural network.
                experiment_name (str): The name of the experiment to be run.
            
            Params:
                n_vars (int): The number of variables in the neural network.
                dom_u (int): The upper limit for weights and biases.
                dom_l (int): The lower limit for weights and biases.
                npop (int): The population size.
                gens (int): The number of generations.
                mutation (float): The mutation rate.
                last_best (int): The last best fitness score.

        """
        # Initialize input arguments
        self.env = env
        self.run_mode = mode
        self.n_hidden_neurons = n_hidden_neurons
        self.experiment_name = experiment_name

        # Initialize genetic algorithm parameters
        self.n_vars = (self.env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
        self.dom_u = 1
        self.dom_l = -1
        self.npop = 100
        self.gens = 30
        self.selection_param = 0.2
        self.crossover_param = 0.2
        self.mutation_param = 0.2

        # Keep track of the population and fitness
        self.pop = None
        self.fit_pop = None
        self.last_best = 0

    def initialize_population(self):
        """
            Initializes the population from scratch or loads from previous state.
        """
        if not os.path.exists(self.experiment_name+'/evoman_solstate'):
            print("Optimizing from scratch.")
            # Initialize population
            self.pop = np.random.uniform(self.dom_l, self.dom_u, (self.npop, self.n_vars))
            # Evaluate fitness
            self.evaluate_fitness()
            # Save initial state
            self.save_results('INITIAL')
        else:
            print("Optimizing from previous state.")
            # Load previous state
            self.env.load_state()
            # Load state's population and fitness
            self.pop = self.env.solutions[0]
            self.fit_pop = self.env.solutions[1]
            
            # Find last generation number
            with open(os.path.join(self.experiment_name, 'gen.txt'), 'r') as file_aux:
                ini_g = int(file_aux.readline())
            self.save_results(ini_g)

    def evaluate_fitness(self):
        """
            Evaluates the fitness of each individual in the population.
        """
        self.fit_pop = np.array([self.env.play(pcont=p)[0] for p in self.pop])

    def selection(self):
        """
            Selects parents for reproduction using tournament selection.
        """
        num_parents = int(self.npop * self.selection_param)
        parents = []
        
        for _ in range(num_parents):
            # Randomly select two individuals for the tournament
            p1, p2 = np.random.choice(self.npop, size=2, replace=False)
            # Select the best individual
            if self.fit_pop[p1] > self.fit_pop[p2]:
                parents.append(self.pop[p1])
            else:
                parents.append(self.pop[p2])
        
        return np.array(parents)
    
    def crossover(self, parents):
        """
            Performs crossover on the parents to create offspring.

            Args:
                parents (np.array): The parents to perform crossover on.
        """
        num_offspring = self.npop - int(self.npop * self.crossover_param)
        offspring = []
        
        for _ in range(num_offspring):
            # Select two random parents
            p1, p2 = np.random.randint(0, len(parents), 2)
            # Select a random crossover point
            cross_point = np.random.randint(0, self.n_vars)
            # Create offspring
            offspring.append(np.concatenate((parents[p1][:cross_point], parents[p2][cross_point:])))
        
        return np.array(offspring)
    
    def mutation(self):
        """
        Apply mutation to the population.
        """
        num_mutations = int(self.npop * self.n_vars * self.mutation_param)
        mutation_indices = np.random.choice(self.npop * self.n_vars, size=num_mutations, replace=False)
        
        for i in mutation_indices:
            # Calculate the individual and gene index
            individual_idx = i // self.n_vars
            gene_idx = i % self.n_vars
            # Mutate the gene
            self.pop[individual_idx, gene_idx] = np.random.uniform(self.dom_l, self.dom_u)

    def update_population(self, parents, offspring):
        """
            Updates the population with the new offspring.
        """
        self.pop[: len(parents)] = parents
        self.pop[len(parents) :] = offspring
        self.evaluate_fitness()

    def save_results(self, i):
        """
            Saves the results of the current generation.

            Args:
                i (int): The current generation number.
                pop (list): The population.
                fit_pop (list): The fitness scores of the population.            
        """
        # Find stats
        best = np.argmax(self.fit_pop)
        mean = np.mean(self.fit_pop)
        std  =  np.std(self.fit_pop)

        # Saves stats for current generation
        with open(os.path.join(self.experiment_name, 'optimization_logs.txt'), 'a') as file_aux:
            print('--------------------------------------')
            print('GENERATION ' + str(i))
            print('best mean std')
            print(str(round(self.fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
            file_aux.write('\n' + str(i) + ' ' + str(round(self.fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))

        # Saves generation number
        with open(os.path.join(self.experiment_name, 'gen.txt'), 'w') as file_aux:
            file_aux.write(str(i))
        
        # Save file with the best solution
        np.savetxt(self.experiment_name+'/best_solution.txt', self.pop[best])

        # Saves simulation state
        solutions = [self.pop, self.fit_pop]
        self.env.update_solutions(solutions)
        self.env.save_state()

    def run(self):
        """
            Runs the Genetic Algorithm.
        """
        # Initialize population and fitness
        self.initialize_population()

        for gen in range(self.gens):
            # Select parents
            parents = self.selection()

            # Crossover
            offspring = self.crossover(parents)

            # Mutation
            self.mutation()

            # Update population and fitness
            self.update_population(parents, offspring)

            # Save/print best individual
            self.save_results(gen)