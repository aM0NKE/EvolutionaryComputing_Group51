# Genetic Algorithm
## Contents:
- `\Results`: contains all of the results generated for the final report.
- `\evoman`: contains the EvoMan framework.
- `Controller.py`: contains neural network connected to the game controller we are aiming to optimize.
- `Experiment.py`: runs the experiment of optimizing the controller for a given enemy for x amount of trials.
- `FindBestConfig.py`: script that finds the best parameter values from a folder containing all parameter tuning results.
- `Optimization.py`: our own implementation of the *Genetic Algorithm*.
- `ParameterTuning.py`: uses a grid search approach to find the best parameter values for: $\lambda$, $k$, and population size.
- `PlotResults.py`: generates the fitness and gain plots over all of the optimization runs. 
- `ShowSolution.py`: runs the game and shows the best-found solution.
## Example Usage:
### Step 1: Parameter Tuning
- Command: `python3 ParameterTuning.py`
- Note: this script takes around 3 to 5 hours to run, depending on the hardware you run it on.
- Find best parameter values: `FindBestConfig.py`
### Step 2: Optimization 
- Command: `python3 Experiment.py -d GA1_EGALL 
- Flags:
 	- `-d`:	directory you want to store the results
	- `-t`: number of optimization trials (Default=10)
  - `-g`: number of generations per optimization cycle (default=30)
- Plot results: `python3 PlotResults.py`
	- Note: make sure all directories and enemy groups are set correctly in the script. 
### Step 4: Show Solution
- Command: `python3 ShowSolution.py -e GA1_EGALL`
- Flags:
	- `-e`: directory to the experiment
	- `-n`: number of neurons (Default=10)
