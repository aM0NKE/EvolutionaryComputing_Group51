#########################################################################################         
#           This code is a reworked version of the "DEMO : Neuroevolution -             #
#       Genetic Algorithm neural network." that was provided by the course admins       #
#                           and written by Karine Miras                                 #
#########################################################################################

# Import framework
import sys 
from evoman.environment import Environment
from demo_controller import player_controller

# Import other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

#########################################################################################
#                                [THE GENETIC ALGORITHM]                                #
#   Optimization for controller solution (best genotype-weights for phenotype-network)  #
#########################################################################################
class Genetic(object):
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
        self.mutation = 0.2
        self.last_best = 0

    def simulation(self, x):
        """
            Runs the simulation for a given set of weights and biases.
        
            Args:
                x (list): The weights and biases to be used in the simulation.
            
            Returns:
                f (float): The fitness score of the simulation.
                p (float): The player life of the simulation.
                e (float): The enemy life of the simulation.
                t (float): The time of the simulation.
        """
        f, p, e, t = self.env.play(pcont=x)
        return f
    
    def norm(self, x, pfit_pop):
        """
            Normalizes the fitness score of a given individual.

            Args:
                x (float): The fitness score of the individual.
                pfit_pop (list): The fitness scores of the population.
            
            Returns:
                x_norm (float): The normalized fitness score of the individual.
        """
        if (max(pfit_pop) - min(pfit_pop)) > 0:
            x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
        else:
            x_norm = 0

        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm
    
    def evaluate(self, x):
        """
            Evaluates the fitness score of a given individual.

            Args:
                x (list): The weights and biases of the individual.

            Returns:
                np.array(list): The fitness score of the individual.
        """
        return np.array(list(map(lambda y: self.simulation(y), x)))

    def tournament(self, pop, fit_pop):
        """
            Selects the best individual from a random sample of the population.

            Args:
                pop (list): The population.
                fit_pop (list): The fitness scores of the population.
        """
        c1 = np.random.randint(0, pop.shape[0], 1)
        c2 = np.random.randint(0, pop.shape[0], 1)

        if fit_pop[c1] > fit_pop[c2]:
            return pop[c1][0]
        else:
            return pop[c2][0]
        
    def limits(self, x):
        """
            Limits the value of a given variable to the upper and lower limits.

            Args:
                x (float): The value to be limited.
        """
        if x > self.dom_u:
            return self.dom_u
        elif x < self.dom_l:
            return self.dom_l
        else:
            return x

    def crossover(self, pop, fit_pop):
        """
            Performs crossover on the population.

            Args:
                pop (list): The population.
                fit_pop (list): The fitness scores of the population.
            
            Returns:
                total_offspring (list): The offspring of the population.
        """
        total_offspring = np.zeros((0, self.n_vars))

        for p in range(0, pop.shape[0], 2):
            p1 = self.tournament(pop, fit_pop)
            p2 = self.tournament(pop, fit_pop)

            n_offspring = np.random.randint(1, 3+1, 1)[0]
            offspring = np.zeros((n_offspring, self.n_vars))

            for f in range(0, n_offspring):

                cross_prop = np.random.uniform(0, 1)
                offspring[f] = p1 * cross_prop + p2 * (1 - cross_prop)

                # mutation
                for i in range(0, len(offspring[f])):
                    if np.random.uniform(0 ,1) <= self.mutation:
                        offspring[f][i] = offspring[f][i] + np.random.normal(0, 1)

                offspring[f] = np.array(list(map(lambda y: self.limits(y), offspring[f])))

                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring

    def doomsday(self, pop, fit_pop):
        """
            Performs doomsday extinction event on the population.
            Thus, kills the worst genomes, and replace with new best/random solutions.

            Args:
                pop (list): The population.
                fit_pop (list): The fitness scores of the population.
        """
        worst = int(self.npop / 4)  # a quarter of the population
        order = np.argsort(fit_pop)
        orderasc = order[0:worst]

        for o in orderasc:
            for j in range(0, self.n_vars):
                pro = np.random.uniform(0, 1)
                if np.random.uniform(0, 1) <= pro:
                    pop[o][j] = np.random.uniform(self.dom_l, self.dom_u) # random dna, uniform dist.
                else:
                    pop[o][j] = pop[order[-1:]][0][j] # dna from best

            fit_pop[o] = self.evaluate([pop[o]])

        return pop, fit_pop
    
    def check_mode(self):
        """
            Checks the mode the simulation is running in.
            If the mode is 'test', the best solution is loaded and run.
        """
        if run_mode == 'test':

            bsol = np.loadtxt(self.experiment_name+'/best.txt')
            print( '\n RUNNING SAVED BEST SOLUTION \n')
            self.env.update_parameter('speed','normal')
            self.evaluate([bsol])

            sys.exit(0)
    
    def load_population(self):
        """
            Initializes population loading old solutions or generating new ones
        """
        if not os.path.exists(experiment_name+'/evoman_solstate'):

            print( '\nNEW EVOLUTION\n')

            pop = np.random.uniform(self.dom_l, self.dom_u, (self.npop, self.n_vars))
            fit_pop = self.evaluate(pop)
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            ini_g = 0
            solutions = [pop, fit_pop]
            self.env.update_solutions(solutions)
        else:

            print( '\nCONTINUING EVOLUTION\n')

            self.env.load_state()
            pop = env.solutions[0]
            fit_pop = env.solutions[1]

            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)

            # finds last generation number
            file_aux  = open(experiment_name+'/gen.txt','r')
            ini_g = int(file_aux.readline())
            file_aux.close()

        # saves results for first pop
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n\ngen best mean std')
        print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        return pop, fit_pop, best, mean, std, ini_g

    def evolution(self, pop, fit_pop, best, mean, std, ini_g):
        """
            Runs the evolution of the population.

            Args:
                pop (list): The population.
                fit_pop (list): The fitness scores of the population.
                best (int): The index of the best individual in the population.
                mean (float): The mean fitness score of the population.
                std (float): The standard deviation of the fitness scores of the population.
                ini_g (int): The initial generation number.
        """
        last_sol = fit_pop[best]
        notimproved = 0

        for i in range(ini_g+1, self.gens):

            offspring = self.crossover(pop, fit_pop)  # crossover
            fit_offspring = self.evaluate(offspring)   # evaluation
            pop = np.vstack((pop,offspring))
            fit_pop = np.append(fit_pop,fit_offspring)

            best = np.argmax(fit_pop) #best solution in generation
            fit_pop[best] = float(self.evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
            best_sol = fit_pop[best]

            # selection
            fit_pop_cp = fit_pop
            fit_pop_norm =  np.array(list(map(lambda y: self.norm(y, fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
            probs = (fit_pop_norm) / (fit_pop_norm).sum()
            chosen = np.random.choice(pop.shape[0], self.npop , p=probs, replace=False)
            chosen = np.append(chosen[1:], best)
            pop = pop[chosen]
            fit_pop = fit_pop[chosen]


            # searching new areas

            if best_sol <= last_sol:
                notimproved += 1
            else:
                last_sol = best_sol
                notimproved = 0

            if notimproved >= 15:

                file_aux  = open(self.experiment_name+'/results.txt','a')
                file_aux.write('\ndoomsday')
                file_aux.close()

                pop, fit_pop = self.doomsday(pop, fit_pop)
                notimproved = 0

            best = np.argmax(fit_pop)
            std  =  np.std(fit_pop)
            mean = np.mean(fit_pop)

            # saves results
            file_aux  = open(self.experiment_name+'/results.txt','a')
            print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
            file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
            file_aux.close()

            # saves generation number
            file_aux  = open(self.experiment_name+'/gen.txt','w')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(self.experiment_name+'/best.txt',pop[best])

            # saves simulation state
            solutions = [pop, fit_pop]
            self.env.update_solutions(solutions)
            self.env.save_state()


    def main(self):
        """
            Runs the Genetic Algorithm.
        """
        # Check mode
        self.check_mode()

        # Initializes population loading old solutions or generating new ones
        pop, fit_pop, best, mean, std, ini_g = self.load_population()

        # Evolution
        self.evolution(pop, fit_pop, best, mean, std, ini_g)

        fim = time.time() # prints total execution time for experiment
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')

        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()

        env.state_to_log() # checks environment state


#########################################################################################
#                                       [MAIN]:                                         #
#########################################################################################
# Set to true for not using visuals and thus making experiments faster.
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Choose experiment name
experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Choose number of hidden neurons for a neural network with one hidden layer.
n_hidden_neurons = 10

# Initialize game simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)
# [NOTE]: Default environment fitness is assumed for experiment

env.state_to_log() # Checks environment state
        
ini = time.time()  # Sets time marker
 
# Train Genetic Algorithm
run_mode = 'train' # train or test
genetic = Genetic(env, run_mode, n_hidden_neurons, experiment_name)
genetic.main()


#########################################################################################
#                                       [NOTES]:                                        #
#########################################################################################
# CODE SNIPPET FROM evoman/environment.py
# Default fitness function for single solutions
# def fitness_single(self):
#     return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - numpy.log(self.get_time())
# Default fitness function for consolidating solutions among multiple games
# def cons_multi(self,values):
#     return values.mean() - values.std()

# Deafault environment parameters
# experiment_name='test',
# multiplemode="no",           # yes or no
# enemies=[1],                 # array with 1 to 8 items, values from 1 to 8
# loadplayer="yes",            # yes or no
# loadenemy="yes",             # yes or no
# level=2,                     # integer
# playermode="ai",             # ai or human
# enemymode="static",          # ai or static
# speed="fastest",             # normal or fastest
# inputscoded="no",            # yes or no
# randomini="no",              # yes or no
# sound="off",                  # on or off
# contacthurt="player",        # player or enemy
# logs="on",                   # on or off
# savelogs="yes",              # yes or no
# clockprec="low",
# timeexpire=3000,             # integer
# overturetime=100,            # integer
# solutions=None,              # any
# fullscreen=False,            # True or False
# player_controller=None,      # controller object
# enemy_controller=None,      # controller object
# use_joystick=False,
# visuals=False