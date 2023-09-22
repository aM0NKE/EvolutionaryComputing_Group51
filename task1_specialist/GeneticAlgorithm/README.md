# Genetic Algorithm
## Contents:
- `Controller.py`: contains neural network connected to game controller we are aiming to optimize.
- `Experiment.py`: runs the experiment of optimizing the controller.
- `FitnessSensitivityAnalysis.py`: runs the sensitiviy analysis w.r.t. gamma and beta for a particular enemy, and plots the results.
- `GeneticOptimization.py`: a reworked version of the *Genetic Algorithm* that was provided by the course admins.
- `GeneticOptimizationV2.py`: our own implementation of the *Genetic Algorithm*.
- `PlotResults.py`: generates the fitness and gain plots over all optimization runs. 
- `ShowSolution.py`: runs game and shows the best found solution.
## Example Usage:
### Step 1: Sensitivity Analysis
- Command: `python3 FitnessSensitivityAnalysis.py -e SA_enemy_2`
- Flags:
	- `-p`: if this flag is set, plot SA results. else, run the experiment. 
 	- `-e`:	directory you want to store the results (make sure enemy_x is in the name)
  	- `-n`: number of neurons
  	- `-t`: number of optimization trials
### Step 2: Optimization 
- Command: `python3 Experiment.py -e 2 -n 10 -t 10 -g 20`
- Flags:
	- `-e`: number corresponding to Evoman enemy (1-8)
	- `-n`: number of neurons
	- `-t`: number of optimization trials
 	- `-g`: number of generations per optimization trial
### Step 3: Plot Results
- Command: `python3 PlotResults.py -e genetic_v_enemy_2/ -s`
- Flags: 
	- `-e`: directory to the experiment
 	- `-s`: if this flag is set, save the plots. else, show the plots. 
### Step 4: Show Solution
- Command: `python3 ShowSolution.py -e genetic_v_enemy_2/`
	- Flags:
		- `-e`: directory to the experiment
		- `-n`: number of neurons
