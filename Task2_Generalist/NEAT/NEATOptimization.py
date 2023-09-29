# Import framework
import sys 
from evoman.environment import Environment
from NEATController import NEATPlayerController

# Import other libs
import neat
import numpy as np
import pickle
import os

class NEATOptimization(object):
    """
        This class implements the NEAT algorithm.
    """
    def __init__(self, env, experiment_name, gens):
        """
            Initializes the NEAT algorithm.

            Args:
                env (Environment): The environment to be used.
                experiment_name (str): The name of the experiment.
                gens (int): The number of generations to run NEAT for.
                config (str): The path to the NEAT config file.
        """
        # Save input parameters
        self.env = env
        self.experiment_name = experiment_name
        self.gens = gens
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                'NEATConfig')

        # Keep track of the population and fitness
        self.pop = None
        self.fit_pop = None
        self.gain_pop = None

    def _initialize_population(self):
        """
            Initializes the population.
        """
        # Initialize NEAT Population object
        self.pop = neat.Population(self.config)

        # Add reporters to show progress in the terminal.
        self.pop.add_reporter(neat.StdOutReporter(True))
        # self.pop.add_reporter(neat.Checkpointer(5))
        stats = neat.StatisticsReporter()
        self.pop.add_reporter(stats)

        # Save initial optimization logs 
        self._save_results(init=True)

    def _eval_genome(self, p):
        """
            Calculates the fitness of a single genome.

            Args:
                p (list): The genome to be evaluated.
        """
        # Run the game
        fitness, player_hp, enemy_hp, time = self.env.play(pcont=p)

        # Calculate gain
        gain = player_hp - enemy_hp

        return fitness, gain

    def _eval_genomes(self, genomes, config):
        """ 
            Evaluates the population of solutions.
        """
        # Reset fitness and gain lists
        self.fit_pop = []
        self.gain_pop = []

        # Evaluate each genome in the population
        for genome_id, genome in genomes:
            genome.fitness, genome.gain = self._eval_genome(genome)
            self.fit_pop.append(genome.fitness)
            self.gain_pop.append(genome.gain)
        
        # Save results to optimization logs
        self._save_results()

    def _save_results(self, init=False):
        """
            Saves the stats of the current generation to optimization logs.

            Args:
                init (bool): Whether this is the first generation.
        """
        # Find stats
        if not init: 
            best_fit = np.argmax(self.fit_pop)
            mean_fit = np.mean(self.fit_pop)
            std_fit  =  np.std(self.fit_pop)
            best_gain = np.argmax(self.gain_pop)
            mean_gain = np.mean(self.gain_pop)
            std_gain = np.std(self.gain_pop)

        # Save stats to optimization logs
        with open(os.path.join(self.experiment_name, 'optimization_logs.txt'), 'a') as file_aux:
            if init: file_aux.write('mean_fit std_fit max_fit mean_gain std_gain max_gain')
            else: file_aux.write('\n' + str(round(mean_fit, 6)) + ' ' + str(round(std_fit, 6)) + ' ' + str(round(self.fit_pop[best_fit], 6)) + ' ' + str(round(mean_gain, 6)) + ' ' + str(round(std_gain, 6)) + ' ' + str(round(self.gain_pop[best_gain], 6)))
        
    def _run(self):
        """
            Runs the NEAT algorithm for a single trial.
        """
        # Initialize population
        self._initialize_population()

        # Run NEAT to find best solution
        best_sol = self.pop.run(self._eval_genomes, self.gens)

        # Save best solution
        with open(self.experiment_name + '/best_solution', 'wb') as f:
            pickle.dump(best_sol, f)
