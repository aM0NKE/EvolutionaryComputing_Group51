#########################################################################################         
#                        This is our implementation of the:                             #
#                Genetic Algorithm for Optimizing a Neural Network.                     #
#           (The controller -that contains the NN- is demo_controller.py)               #
#########################################################################################

# Import framework
import sys 
import glob, os
from evoman.environment import Environment
from Controller import GeneticController

# Import other libs
import numpy as np
import random

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
        experiment_name (str): The name of the experiment to be run.
        n_hidden_neurons (int): The number of hidden neurons in the neural network.
        dom_u (int): The upper limit for weights and biases.
        dom_l (int): The lower limit for weights and biases.
        n_pop (int): The population size.
        generations (int): The max number of generations.
        gamma (float): Fitness func. weight for the enemy damage.
        alpha (float): The weight for the player damage.
        selection_lambda (int): The number of parents to select.
        selection_k (int): The number of individuals to select for the tournament.
        crossover_alpha (float): The crossover parameter.
        mutation_rate (float): Percentage of genes to mustate.
    """

    def __init__(self, 
                 env, 
                 experiment_name, 
                 n_hidden_neurons=10, 
                 dom_u=1, 
                 dom_l=-1, 
                 n_pop=50, 
                 generations=25, 
                 gamma=0.9, 
                 alpha=0.1,
                 selection_lambda=12,
                 selection_k=4,
                 crossover_alpha=0.5,
                 mutation_rate=0.1):
        """
            Initializes the Genetic Algorithm.
            
            Attributes:
                n_vars (int): The number of variables in the neural network.
                pop (np.array): The population of individuals.
                best_fitness (float): The best fitness found so far.
                fitness_values (list): The fitness values of the current population.
                gain_values (list): The gain values of the current population.
        """
        # Initialize input arguments
        self.env = env
        self.experiment_name = experiment_name

        # Initialize Population
        self.n_pop = n_pop
        self.dom_u = dom_u
        self.dom_l = dom_l
        self.n_hidden_neurons = n_hidden_neurons
        self.n_vars = (self.env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
        self.pop = np.random.uniform(self.dom_l, self.dom_u, (self.n_pop, self.n_vars))

        # Generation params
        self.generations = generations
        self.best_fitness = float('-inf')
        self.fitness_values = np.array([])
        self.gain_values = np.array([])
        self._evaluate_population()
        self.save_results(0)
        
        # Fitness func weights
        self.gamma = gamma
        self.alpha = alpha
        
        # Selection params
        self.selection_lambda = selection_lambda
        self.selection_k = selection_k
        
        # Crossover param
        self.crossover_alpha = crossover_alpha
        
        # Mutation param
        self.mutation_rate = mutation_rate

    def _fitness_function(self, enemylife, playerlife, time):
        """
            Calculates the fitness value for an individual solution.

            Args:
                enemylife (float): The enemy's life.
                playerlife (float): The player's life.
                time (float): The time it took to finish the game.
        """
        return self.gamma * (100 - enemylife) + self.alpha * playerlife - np.log(time)

    def _evaluate_solution(self, p):
        """
            Calculates the fitness and gain of a single genome.

            Args:
                p (list): The genome to be evaluated.
        """
        # Run the game
        vfitness, vplayerlife, venemylife, vtime = self.env.play(pcont=p)

        # Calculate gain
        vgain = vplayerlife - venemylife
        return vfitness, vgain
    
    def _evaluate_population(self):
        """
            Evaluates the fitness of the current population.
        """
        self.fitness_values = np.array([])
        self.gain_values = np.array([])
        for individual in self.pop:
            fitness, gain = self._evaluate_solution(individual)
            self.fitness_values = np.append(self.fitness_values, fitness)
            self.gain_values = np.append(self.gain_values, gain)
    
    def _tournament_selection(self):
        """
            Selects parents using tournament selection.

            Params:
                selection_lambda (int): The number of parents to select.
                selection_k (int): The number of individuals to select for the tournament.
        """
        selected_parents = []
        for _ in range(self.selection_lambda):
            # Select individuals for tournament
            tournament_indices = np.random.choice(self.n_pop, size=self.selection_k, replace=False)
            tournament = self.pop[tournament_indices]
            # Evaluate fitness of individuals in tournament
            tournament_fitness = self.fitness_values[tournament_indices]
            # Select best individual and add to parent pool
            best_index = np.argmax(tournament_fitness)
            selected_parents.append(tournament[best_index])
        return selected_parents
    
    def _blend_crossover(self, parents):
        """
            Performs Blend Crossover on two parents.

            Args:
                parents (list): The parents to perform crossover on.
            
            Param:
                crossover_alpha (float): The crossover parameter.
        """
        gamma = (1 - 2 * self.crossover_alpha) * np.random.uniform(0, 1, size=self.n_vars) - self.crossover_alpha
        parent1, parent2 = parents[0], parents[1]
        child = (1 - gamma) * parent1 + gamma * parent2
        return child
    
    def _nonuniform_mutation(self, child):
        """
            Performs Nonuniform Mutation on a child.

            Args:
                child (np.array): The child to perform mutation on.
        
            Param:
                mutation_rate (float): Percentage of genes to mustate.
        """
        mutated_child = np.copy(child)
        # Select genes to mutate
        mutation_mask = np.random.uniform(0, 1, size=self.n_vars) < self.mutation_rate
        # Mutate genes
        mutated_child[mutation_mask] += np.random.uniform(-0.1, 0.1, size=np.sum(mutation_mask))
        return mutated_child
    
    def _replace_worst(self, children):
        """
            Adds children to the population and removes the worst individuals.

            Args:
                gen (int): The current generation.
                children (list): The children to be added to the population.
        """
        # Add children to population
        self.pop = np.vstack((self.pop, np.array(children)))
        # Evaluate and rank population
        self._evaluate_population()
        sorted_indices = np.argsort(self.fitness_values)[::-1]
        # Remove worst individuals
        self.pop = self.pop[sorted_indices][:self.n_pop]
        self.fitness_values = self.fitness_values[sorted_indices][:self.n_pop]
        self.gain_values = self.gain_values[sorted_indices][:self.n_pop]

    def save_results(self, gen):
        # Calculate stats
        mean_fit = np.mean(self.fitness_values)
        std_fit = np.std(self.fitness_values)
        max_fit = np.max(self.fitness_values)
        mean_gain = np.mean(self.gain_values)
        std_gain = np.std(self.gain_values)
        max_gain = np.max(self.gain_values)
        
        # Print training stats
        print('gen: {} | mean_fit: {} | std_fit: {} | max_fit: {} | mean_gain: {} | std_gain: {} | max_gain: {}'.format(gen, str(round(mean_fit, 6)), str(round(std_fit, 6)), str(round(max_fit, 6)), str(round(mean_gain, 6)), str(round(std_gain, 6)), str(round(max_gain, 6))))
        
        # Save training logs
        with open(os.path.join(self.experiment_name, 'optimization_logs.txt'), 'a') as file_aux:
            if gen == 0: 
                file_aux.write('gen mean_fit std_fit max_fit mean_gain std_gain max_gain')
                file_aux.write('\n{} {} {} {} {} {} {}'.format(gen, str(round(mean_fit, 6)), str(round(std_fit, 6)), str(round(max_fit, 6)), str(round(mean_gain, 6)), str(round(std_gain, 6)), str(round(max_gain, 6))))
            else:   
                file_aux.write('\n{} {} {} {} {} {} {}'.format(gen, str(round(mean_fit, 6)), str(round(std_fit, 6)), str(round(max_fit, 6)), str(round(mean_gain, 6)), str(round(std_gain, 6)), str(round(max_gain, 6))))    

        # Save population of the current generation
        if gen == 0:
            os.makedirs(os.path.join(self.experiment_name, 'generations'))
        generation_file_path = os.path.join(self.experiment_name, 'generations/gen_{}.txt'.format(gen))
        with open(generation_file_path, 'a') as file_aux:
            np.savetxt(file_aux, self.pop)

        # Save generation number
        with open(os.path.join(self.experiment_name, 'gen_num.txt'), 'w') as file_aux:
            file_aux.write(str(gen))

    def optimize(self):
        """
            Finds the best solution for EvoMan.
        """
        for gen in range(1, self.generations + 1):
            print("\nGeneration: {}".format(gen))

            # Select parents
            parents = self._tournament_selection()

            # Generate children
            children = [self._blend_crossover(random.sample(parents, 2)) for _ in range(self.selection_lambda)]

            # Mutate children
            children = [self._nonuniform_mutation(child) for child in children]

            # Replacement
            self._replace_worst(children)

            # Check if the best solution has improved
            current_best_fitness = self.fitness_values[0]
            if current_best_fitness > self.best_fitness:
                np.savetxt(os.path.join(self.experiment_name, 'best_solution.txt'), self.pop[0])
                self.best_fitness = current_best_fitness

            # Save results
            self.save_results(gen)

        return self.pop[0], self.best_fitness