# Import libs
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt


def parse_arguments():
    """
        Parses the input arguments.
        
        Args:
            enemy (int): The enemy to be used to train the neural network.
            n_hidden_neurons (int): The number of hidden neurons in the neural network.
            visuals (bool): Whether to use visuals or not.
            mode (str): The mode to run the simulation in. Either 'train' or 'test'.

    """
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add input arguments
    parser.add_argument('-e', '--experiment_name', type=str, help='Directory to experiment')

    return parser.parse_args()

if __name__ == "__main__":

    # Parse input arguments
    args = parse_arguments()

    all_mean_fit = []
    all_std_fit = []
    all_max_fit = []

    # For each trial
    trials = glob.glob(args.experiment_name + '/*')
    for t in trials:
        # Get mean, std, max fitness over all trials
        data = pd.read_csv(t + '/optimization_logs.txt', sep='\s+', header=0)
        all_mean_fit.append(data['mean_fit'])
        all_std_fit.append(data['std_fit'])
        all_max_fit.append(data['max_fit'])

    # Get data for plotting
    avg_mean_fit = [sum(x)/len(x) for x in zip(*all_mean_fit)]
    avg_std_fit = [sum(x)/len(x) for x in zip(*all_std_fit)]
    avg_max_fit = [sum(x)/len(x) for x in zip(*all_max_fit)]
    min_mean_fit = [min(x) for x in zip(*all_mean_fit)]
    max_mean_fit = [max(x) for x in zip(*all_mean_fit)]
    min_max_fit = [min(x) for x in zip(*all_max_fit)]
    max_max_fit = [max(x) for x in zip(*all_max_fit)]

    print(all_mean_fit)
    print(min_mean_fit)

    # Draw lines
    plt.plot(data['gen'], avg_mean_fit, marker='o', linestyle='-', label='Avg. Mean Fitness', color='blue')
    plt.fill_between(data['gen'], min_mean_fit, max_mean_fit, alpha=0.2, color='blue')
    plt.plot(data['gen'], avg_max_fit, marker='o', linestyle='-', label='Avg. Max Fitness', color='red')
    plt.fill_between(data['gen'], min_max_fit, max_max_fit, alpha=0.2, color='red')

    # Show plot
    plt.title('Genetic v. Enemy ' + args.experiment_name.split('_')[-1][0])
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()







