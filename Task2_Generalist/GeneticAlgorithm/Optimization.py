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
                 n_pop=75, 
                 generations=30, 
                 gamma=0.6, 
                 alpha=0.4,
                 selection_lambda=25,
                 selection_k=8,
                 crossover_alpha=0.5,
                 mutation_rate=0.2, 
                 mutation_step_size=0.1):
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
        self.pop = np.empty((0, self.n_vars))

        # Generation params
        self.generations = generations
        self.best_fitness = float('-inf')
        self.fitness_values = np.array([])
        self.gain_values = np.array([])
        
        # Fitness func weights
        self.gamma = gamma
        self.alpha = alpha
        
        # Selection params
        self.selection_lambda = selection_lambda
        self.selection_k = selection_k
        
        # Crossover param
        self.crossover_alpha = crossover_alpha
        
        # Non-uniform Mutation param
        self.mutation_rate = mutation_rate
        self.mutation_step_size = mutation_step_size

        # Self-adaptive mutation (uncorrelated/step_size=1) params
        # self.tau = 1 / np.sqrt(self.n_vars)
        # self.sigma = np.exp(self.tau * np.random.normal(0, 1, size=self.n_vars))

        # Self-adaptive mutation (uncorrelated/step_size=n) params
        self.tau_prime = 1 / np.sqrt(2 * self.n_vars)
        self.tau = 1 / np.sqrt(2 * np.sqrt(self.n_vars))
        self.sigmas = np.ones((self.n_pop, self.n_vars))
        self.children_sigmas = np.array([])

        self.initilize_population()
        self._evaluate_population()
        self.save_results(0)

    def initilize_population(self):
        """
            Initializes the population with previous solution.
        """

        # find files in folder /load_from
        solutions = glob.glob('load_from/*.txt')
        for file in solutions:
            # open solution
            solution = np.loadtxt(file).reshape(1, self.n_vars)
            self.pop = np.vstack((self.pop, solution))

        while self.pop.shape[0] < self.n_pop:
            # Initialize population
            self.pop = np.vstack((self.pop, np.random.uniform(self.dom_l, self.dom_u, size=self.n_vars)))

    def _fitness_function(self, enemylife, playerlife, time):
        """
            Calculates the fitness value for an individual solution.

            Args:
                enemylife (float): The enemy's life.
                playerlife (float): The player's life.
                time (float): The time it took to finish the game.
        """
        return self.gamma * (100 - enemylife) + self.alpha * playerlife

    def _evaluate_solution(self, p):
        """
            Calculates the fitness and gain of a single genome.

            Args:
                p (list): The genome to be evaluated.
        """
        # Run the game
        _, vplayerlife, venemylife, vtime = self.env.play(pcont=p)

        # print(self.env.player_controller.shot_cnt/4)
        # self.env.player_controller.shot_cnt = 0

        vfitness = self._fitness_function(venemylife, vplayerlife, vtime)
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
        
        # Rank population
        sorted_indices = np.argsort(self.fitness_values)[::-1]
        self.pop = self.pop[sorted_indices]
        self.fitness_values = self.fitness_values[sorted_indices]
        self.gain_values = self.gain_values[sorted_indices]
        self.sigmas = self.sigmas[sorted_indices]
    
    def _tournament_selection(self):
        """
            Selects parents using tournament selection.

            Params:
                selection_lambda (int): The number of parents to select.
                selection_k (int): The number of individuals to select for the tournament.
        """
        selected_parents, parents_ids = [], []
        for _ in range(self.selection_lambda):
            # Select individuals for tournament
            tournament_indices = np.random.choice(self.n_pop, size=self.selection_k, replace=False)
            tournament = self.pop[tournament_indices]
            # Evaluate fitness of individuals in tournament
            tournament_fitness = self.fitness_values[tournament_indices]
            # Select best individual and add to parent pool
            best_index = np.argmax(tournament_fitness)
            selected_parents.append(tournament[best_index])
            parents_ids.append(tournament_indices[best_index])
        return selected_parents, parents_ids
    
    def _blend_crossover(self, parents_ids):
        """
            Performs Blend Crossover on two parents.

            Args:
                parents (list): The parents to perform crossover on.
            
            Param:
                crossover_alpha (float): The crossover parameter.
        """
        children = []
        children_sigmas_list = []  # Initialize as a list
        for i in range(self.selection_lambda):
            gamma = (1 - 2 * self.crossover_alpha) * np.random.uniform(0, 1, size=self.n_vars) - self.crossover_alpha
            parents_ids = random.sample(parents_ids, 2)
            parent1, parent2 = self.pop[parents_ids[0]], self.pop[parents_ids[1]]
            children.append((1 - gamma) * parent1 + gamma * parent2)
            children_sigmas_list.append(np.abs((1 - gamma) * self.sigmas[parents_ids[0]] + gamma * self.sigmas[parents_ids[1]]))
        # Convert the list to a NumPy array
        self.children_sigmas = np.array(children_sigmas_list)
        return children
    
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
        mutated_child[mutation_mask] += np.random.uniform(-self.mutation_step_size, self.mutation_step_size, size=np.sum(mutation_mask))
        return mutated_child
    
    def _self_adaptive_mutation_one(self, children):
        """
            Performs Self-Adaptive Mutation (uncorrelated and mutation step size one) on 
            a set of children.

            Args:
                children (list): The children to be mutated.
        """
        # Update sigma
        self.sigma = self.sigma * np.exp(self.tau * np.random.normal(0, 1, size=self.n_vars))
        # Boundary rule used to force step sizes to be no smaller than a threshold
        self.sigma[self.sigma < 1e-10] = 1e-10
        # Mutate children
        mutated_children = np.copy(children)
        mutated_children += self.sigma * np.random.normal(0, 1, size=(self.selection_lambda, self.n_vars))
        return mutated_children
    
    def _self_adaptive_mutation_n(self, children):
        """
            Performs Self-Adaptive Mutation (uncorrelated and mutation step size n) on 
            a set of children.

            Args:
                children (list): The children to be mutated.
        """
        # Update sigmas
        self.sigmas = self.sigmas * np.exp(self.tau_prime * np.random.normal(0, 1, size=(self.n_pop, self.n_vars)) + self.tau * np.random.normal(0, 1, size=(self.n_pop, self.n_vars)))
        self.children_sigmas = self.children_sigmas * np.exp(self.tau_prime * np.random.normal(0, 1, size=(self.selection_lambda, self.n_vars)) + self.tau * np.random.normal(0, 1, size=(self.selection_lambda, self.n_vars)))
        # Boundary rule used to force step sizes to be no smaller than a threshold
        self.sigmas[self.sigmas < 1e-10] = 1e-10
        self.children_sigmas[self.children_sigmas < 1e-10] = 1e-10
        # Mutate children
        mutated_children = np.copy(children)
        mutated_children += self.children_sigmas * np.random.normal(0, 1, size=(self.selection_lambda, self.n_vars))
        return mutated_children

    def _replace_worst(self, parents_ids, children):
        """
            Adds children to the population and removes the worst individuals.

            Args:
                gen (int): The current generation.
                children (list): The children to be added to the population.
        """
        n_children = len(children)
        
        # Remove last n_children individuals from population
        self.pop = self.pop[:-n_children]

        # Add children to population
        self.pop = np.vstack((self.pop, np.array(children)))[:self.n_pop]

        # Evaluate and rank population
        self._evaluate_population()

    def _mu_lambda_selection(self, parents_ids, children):
        """
            Performs (mu, lambda) selection on the population.

            Note: lambda > mu has to hold a.k.a. len children > len parents

            Args:
                parents_ids (list): The parent ids to be removed to the population.
                children (list): The children to be ranked added to the population.
        """
        # Filter duplicate parent ids
        parents_ids = list(set(parents_ids))
        n_parents = len(parents_ids)
        # Remove parents_ids from self.pop
        self.pop = np.delete(self.pop, parents_ids, axis=0)
        self.sigmas = np.delete(self.sigmas, parents_ids, axis=0)
        # Rank children solutions based on fitness and select best children
        children_fitness = [self.env.play(pcont=child)[0] for child in children]
        sorted_indices = np.argsort(children_fitness)[::-1]
        best_children = np.array(children)[sorted_indices][:n_parents]
        best_children_sigmas = np.array(self.children_sigmas)[sorted_indices][:n_parents]
        # Add best children to population
        self.pop = np.vstack((self.pop, best_children))
        self.sigmas = np.vstack((self.sigmas, best_children_sigmas))
        # Evaluate and rank population
        self._evaluate_population()

    def save_results(self, gen):
        # Calculate stats
        mean_fit = np.mean(self.fitness_values)
        std_fit = np.std(self.fitness_values)
        max_fit = np.max(self.fitness_values)
        mean_gain = np.mean(self.gain_values)
        std_gain = np.std(self.gain_values)
        max_gain = np.max(self.gain_values)
        
        # Print training stats
        print('gen: {} | mean_fit: {} | std_fit: {} | max_fit: {} | mean_gain: {} | std_gain: {} | max_gain: {}'.format(gen, str(round(mean_fit, 2)), str(round(std_fit, 2)), str(round(max_fit, 2)), str(round(mean_gain, 2)), str(round(std_gain, 2)), str(round(max_gain, 2))))
        
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
            parents, parents_ids = self._tournament_selection()

            # Generate children
            # children = [self._blend_crossover(random.sample(parents_ids, 2)) for _ in range(self.selection_lambda)]
            children = self._blend_crossover(parents_ids)

            # Mutate children
            # children = [self._nonuniform_mutation(child) for child in children]
            children = self._self_adaptive_mutation_n(children)

            # Replacement
            self._replace_worst(parents_ids, children)
            # self._mu_lambda_selection(parents_ids, children)

            # Check if the best solution has improved
            current_best_fitness = self.fitness_values[0]
            if current_best_fitness > self.best_fitness:
                np.savetxt(os.path.join(self.experiment_name, 'best_solution.txt'), self.pop[0])
                self.best_fitness = current_best_fitness

            # Save results
            self.save_results(gen)

        return self.pop[0], self.best_fitness