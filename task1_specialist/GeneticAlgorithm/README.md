# Genetic Algorithm
## Contents:
- `Controller.py`: contains neural network connected to game controller we are aiming to optimize.
- `Experiment.py`: runs the experiment of optimizing the controller.
- `GeneticOptimization.py`: a reworked version of the *Genetic Algorithm* that was provided by the course admins.
- `GeneticOptimizationV2.py`: our own implementation of the *Genetic Algorithm*.
- `PlotOptimizationResults.py`: generates the fitness and gain plots over all optimization runs. 
- `ShowBestSolution.py`: runs game and shows the best found solution.
## Example Usage:
### Step 1: Run Experiment
- Command: `python3 Experiment.py -e 8 -n 10 -t 5`
- Flags:
	- `-e`: Evoman enemy (1-8)
	- `-n`: number of neurons
	- `-t`: number of optimization trials
### Step 2: Plot Results
- Command: `python3 PlotResults.py -e genetic_v_enemy_8/`
- Flags: 
	- `-e`: directory to the experiment
### Step 3: Show Solution
- Command: `python3 ShowSolution.py -e 8 -n 10`
	- Flags:
		- `-e`: Evoman enemy (1-8)
		- `-n`: number of neurons
