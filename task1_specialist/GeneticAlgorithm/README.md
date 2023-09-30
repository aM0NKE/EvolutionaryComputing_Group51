# Genetic Algorithm
## Contents:
- `\Results`: contains all of the results generated for the final report.
- `\evoman`: contains the EvoMan framework.
- `Controller.py`: contains neural network connected to the game controller we are aiming to optimize.
- `Experiment.py`: runs the experiment of optimizing the controller for a given enemy for x amount of trials.
- `FitnessSensitivityAnalysis.py`: runs the sensitiviy analysis w.r.t. $\gamma$ and $\alpha$ for a particular enemy, and plots the results.
- `GeneticOptimization.py`: our own implementation of the *Genetic Algorithm*.
- `ParameterTuning.py`: uses a grid search approach to find the best parameter values for: $\lambda$, $k$, and mutation rate.
- `PlotBehavior.py`: plots the behavioral comparison between _The Striker_ and _The Ninja_.
- `PlotResults.py`: generates the fitness and gain plots over all of the optimization runs. 
- `PlotSA.py`: plots the results of the Sensitivity Analysis. 
- `ShowSolution.py`: runs the game and shows the best-found solution.
## Example Usage:
### Step 1: Sensitivity Analysis
- Command: `python3 FitnessSensitivityAnalysis.py -e SA_enemy_5`
- Flags:
 	- `-e`:	directory you want to store the results (make sure enemy_x is in the name)
  	- `-t`: number of optimization trials (Default=3)
- Note: this script takes around 3 to 5 hours to run, depending on the hardware you run it on.
- Plot results: `python3 PlotSA.py -e SA_enemy_5`
### Step 2: Parameter Tuning
- Command: `python3 ParameterTuning.py`
- Note: this script takes around 3 to 5 hours to run, depending on the hardware you run it on.
### Step 3: Optimization 
- Command: `python3 Experiment.py -e gamma_0.9_alpha_0.1_enemy_5`
- Flags:
 	- `-e`:	directory you want to store the results (make sure enemy_x/gamma_y/alpha_z are in the name)
	- `-t`: number of optimization trials (Default=10)
- Plot results: `python3 PlotResults.py -e gamma_0.9_alpha_0.1_enemy_5 -s`
	- `-s`: if this flag is set, save the plots. else, show the plots. 
### Step 4: Compare Behavior
- Command: `python3 PlotBehavior.py`
- Note: make sure 'FINALBehavioral_evaluation.csv' (which contains the behavioral evaluation of all trials/enemies) is in the same directory. 
### Step 5: Show Solution
- Command: `python3 ShowSolution.py -e gamma_0.9_alpha_0.1_enemy_5`
- Flags:
	- `-e`: directory to the experiment
	- `-n`: number of neurons (Default=10)
