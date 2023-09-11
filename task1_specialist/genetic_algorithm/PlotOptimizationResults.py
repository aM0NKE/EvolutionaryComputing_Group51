# Import libs
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments():
    """
        Parses the input arguments.
        
        Args:
            experiment_name (str): The name of the experiment to plot the results for.
    """
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add input arguments
    parser.add_argument('-e', '--experiment_name', type=str, help='Directory to experiment')

    return parser.parse_args()

def plot_fitness(experiment_name, data, all_mean_fit, all_std_fit, all_max_fit):

    # Get data for plotting
    avg_mean_fit = [sum(x)/len(x) for x in zip(*all_mean_fit)]
    avg_std_fit = [sum(x)/len(x) for x in zip(*all_std_fit)]
    avg_max_fit = [sum(x)/len(x) for x in zip(*all_max_fit)]
    min_mean_fit = [min(x) for x in zip(*all_mean_fit)]
    max_mean_fit = [max(x) for x in zip(*all_mean_fit)]
    min_max_fit = [min(x) for x in zip(*all_max_fit)]
    max_max_fit = [max(x) for x in zip(*all_max_fit)]
    
    # Draw lines
    plt.plot(data['gen'], avg_mean_fit, marker='o', linestyle='-', label='Avg. Mean Fitness', color='blue')
    plt.fill_between(data['gen'], min_mean_fit, max_mean_fit, alpha=0.2, color='blue')
    plt.plot(data['gen'], avg_max_fit, marker='o', linestyle='-', label='Avg. Max Fitness', color='red')
    plt.fill_between(data['gen'], min_max_fit, max_max_fit, alpha=0.2, color='red')

    # Show plot
    plt.title('Genetic v. Enemy ' + experiment_name.split('_')[-1][0])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()

    # Save plot
    plt.savefig(experiment_name + '/fitness_plot.png')

def plot_gain(experiment_name, data, all_mean_gain, all_std_gain, all_max_gain):

    # Get data for plotting
    avg_mean_gain = [sum(x)/len(x) for x in zip(*all_mean_gain)]
    avg_std_gain = [sum(x)/len(x) for x in zip(*all_std_gain)]
    avg_max_gain = [sum(x)/len(x) for x in zip(*all_max_gain)]
    min_mean_gain = [min(x) for x in zip(*all_mean_gain)]
    max_mean_gain = [max(x) for x in zip(*all_mean_gain)]
    min_max_gain = [min(x) for x in zip(*all_max_gain)]
    max_max_gain = [max(x) for x in zip(*all_max_gain)]
    
    # Draw lines
    plt.plot(data['gen'], avg_mean_gain, marker='o', linestyle='-', label='Avg. Mean Gain', color='blue')
    plt.fill_between(data['gen'], min_mean_gain, max_mean_gain, alpha=0.2, color='blue')
    plt.plot(data['gen'], avg_max_gain, marker='o', linestyle='-', label='Avg. Max Gain', color='red')
    plt.fill_between(data['gen'], min_max_gain, max_max_gain, alpha=0.2, color='red')

    # Show plot
    plt.title('Genetic v. Enemy ' + experiment_name.split('_')[-1][0])
    plt.xlabel('Generation')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()

    # Save plot
    plt.savefig(experiment_name + '/gain_plot.png')

if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    all_mean_fit = []
    all_std_fit = []
    all_max_fit = []
    all_mean_gain = []
    all_std_gain = []
    all_max_gain = []

    # For each trial
    trials = glob.glob(args.experiment_name + '/*')
    for t in trials:
        # Get mean, std, max fitness over all trials
        data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
        all_mean_fit.append(data['mean_fit'])
        all_std_fit.append(data['std_fit'])
        all_max_fit.append(data['max_fit'])
        all_mean_gain.append(data['mean_gain'])
        all_std_gain.append(data['std_gain'])
        all_max_gain.append(data['max_gain'])

    plot_fitness(args.experiment_name, data, all_mean_fit, all_std_fit, all_max_fit)
    plot_gain(args.experiment_name, data, all_mean_gain, all_std_gain, all_max_gain)
