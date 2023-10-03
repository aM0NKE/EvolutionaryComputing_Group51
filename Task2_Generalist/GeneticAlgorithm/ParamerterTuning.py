# Import framework
import sys 
import glob, os
from evoman.environment import Environment
from Task2_Generalist.GeneticAlgorithm.Controller import GeneticController
from Task2_Generalist.GeneticAlgorithm.Optimization import GeneticOptimization

# Import other libs
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd

def mkdir_experiment(experiment_name):
    """
        Sets the experiment name.
    """
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    else:
        os.system('rm -r ' + experiment_name)
        os.makedirs(experiment_name)