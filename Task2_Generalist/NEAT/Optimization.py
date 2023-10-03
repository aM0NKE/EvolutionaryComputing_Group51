# Import framework
import sys 

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
        self.fitness_values = np.array([])
        self.gain_values = np.array([])

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
            Calculates the fitness and gain of a single genome.

            Args:
                p (list): The genome to be evaluated.
        """
        # Run the game
        vfitness, vplayerlife, venemylife, vtime = self.env.play(pcont=p)

        # Calculate gain
        vgain = vplayerlife - venemylife
        return vfitness, vgain

    def _eval_genomes(self, genomes, config):
        """ 
            Evaluates the population of solutions.
        """
        # Reset fitness and gain lists
        self.fitness_values = np.array([])
        self.gain_values = np.array([])

        # Evaluate each genome in the population
        for genome_id, genome in genomes:
            genome.fitness, genome.gain = self._eval_genome(genome)
            self.fitness_values = np.append(self.fitness_values, genome.fitness)
            self.gain_values = np.append(self.gain_values, genome.gain)
        
        # Save results to optimization logs
        self._save_results()

    def _save_results(self, init=False):
        """
            Saves the stats of the current generation to optimization logs.

            Args:
                init (bool): Whether this is the first generation.
        """
        if init:
            initial_eval = [self._eval_genome(genome) for genome_id, genome in self.pop.population.items()]
            self.fitness_values = np.append(self.fitness_values, [x[0] for x in initial_eval])
            self.gain_values = np.append(self.gain_values, [x[1] for x in initial_eval])

        # Find stats
        mean_fit = np.mean(self.fitness_values)
        std_fit  =  np.std(self.fitness_values)
        max_fit = np.max(self.fitness_values)
        mean_gain = np.mean(self.gain_values)
        std_gain = np.std(self.gain_values)
        max_gain = np.max(self.gain_values)

        # Save stats to optimization logs
        with open(os.path.join(self.experiment_name, 'optimization_logs.txt'), 'a') as file_aux:
            if init:
                file_aux.write('mean_fit std_fit max_fit mean_gain std_gain max_gain')
                file_aux.write('\n{} {} {} {} {} {}'.format(str(round(mean_fit, 6)), str(round(std_fit, 6)), str(round(max_fit, 6)), str(round(mean_gain, 6)), str(round(std_gain, 6)), str(round(max_gain, 6))))
            else:   
                file_aux.write('\n{} {} {} {} {} {}'.format(str(round(mean_fit, 6)), str(round(std_fit, 6)), str(round(max_fit, 6)), str(round(mean_gain, 6)), str(round(std_gain, 6)), str(round(max_gain, 6))))    

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
