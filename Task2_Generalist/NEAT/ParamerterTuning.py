# Import framework
import sys 
import glob, os
from evoman.environment import Environment
from Controller import NEATController
from Optimization import NEATOptimization

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