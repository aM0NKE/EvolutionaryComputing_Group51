import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# def add_eval_data_to_file(experiment_name):

#     # Extract enemy    
#     enemy = re.search(r'enemy_(\d+)', args.experiment_name).group(1)

#     # Loop over the trials
#     trials = glob.glob(os.path.join(experiment_name, 'trial*'))
#     for trail_n, t in enumerate(trials):
#         # Initialize lists to store data
#         p_hps, e_hps, times, gains, accs = [], [], [], [], []

#         # Initialize environment
#         env = Environment(experiment_name=t,
#                         enemies=[enemy],
#                         playermode="ai",
#                         player_controller=PlayerController(10),
#                         enemymode="static",
#                         level=2,
#                         speed="fastest",
#                         visuals=False)
        
#         # Loop over all generations in trial
#         gen_tot = int(np.loadtxt(os.path.join(t, 'gen_num.txt')))
#         for gen_n in range(gen_tot):
#             p_hps.append([])
#             e_hps.append([])
#             times.append([])
#             gains.append([])
#             accs.append([])

#             # Load number of gens for particular trial
#             gen_i = np.loadtxt(os.path.join(t, 'generations/gen_' + str(gen_n+1) + '.txt'))

#             # Loop over all solutions in generation
#             for s in range(gen_i.shape[0]):
#                 # Play game
#                 fitness, player_hp, enemy_hp, time = env.play(pcont=gen_i[s])
#                 # Calculate accuracy
#                 accuracy = (100-enemy_hp)/env.player_controller.shot_cnt
#                 accs[gen_n].append(accuracy)
#                 # Calculate gain
#                 gain = player_hp - enemy_hp 
#                 gains[gen_n].append(gain)
#                 # Append data
#                 p_hps[gen_n].append(player_hp)
#                 e_hps[gen_n].append(enemy_hp)
#                 times[gen_n].append(time)

#             accs[gen_n] = np.mean(accs[gen_n])
#             gains[gen_n] = np.mean(gains[gen_n])
#             p_hps[gen_n] = np.mean(p_hps[gen_n])
#             e_hps[gen_n] = np.mean(e_hps[gen_n])
#             times[gen_n] = np.mean(times[gen_n])
        
#         # Save data p_hps, e_hps, times, gains, accs
#         with open(os.path.join(experiment_name, 'behavioral_eval.txt'), 'a') as file_aux:
#             if trail_n == 0: file_aux.write('trial gen p_hp e_hp time gain acc')
#             for gen_n in range(gen_tot):
#                 file_aux.write('\n{} {} {} {} {} {} {}'.format(trail_n, gen_n, round(p_hps[gen_n], 2), round(e_hps[gen_n], 2), round(times[gen_n], 2), round(gains[gen_n], 2), round(accs[gen_n], 2)))

#         # Reset lists
#         p_hps, e_hps, times, gains, accs = [], [], [], [], []

def plot_SA_results(data1, data2, data3):
    # Plot 
    fig, ax = plt.subplots(figsize=(6, 4))

    # Determine the number of bars for each dataset
    num_bars = len(data1)

    # Set the width of the bars to ensure they are next to each other
    bar_width = 0.2

    # Create an array of x positions for the bars
    x1 = np.arange(num_bars)
    x2 = x1 + bar_width
    x3 = x1 + 2 * bar_width

    ax.bar(x1, data1['avg_mean_fit'], width=bar_width, color='#4272f5', alpha=1.0, label='Enemy 5')
    ax.bar(x2, data2['avg_mean_fit'], width=bar_width, color='#f5425a', alpha=1.0, label='Enemy 7')
    ax.bar(x3, data3['avg_mean_fit'], width=bar_width, color='#42f542', alpha=1.0, label='Enemy 8')

    ax.set_xlabel('Gamma/Alpha')
    ax.set_ylabel('Average Mean Fit (over 3 trials)')
    ax.set_title('Sensitivity Analysis - Fitness Function')

    # Set the x-axis tick positions and labels
    ax.set_xticks(x1 + bar_width / 2)
    ax.set_xticklabels(data1['gamma'].astype(str) + '/' + data1['alpha'].astype(str), rotation=45)

    # Add a legend to distinguish between the datasets
    ax.legend(loc='lower right')

    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # Read SA results
    data1 = pd.read_csv("FINALSensitivityAnalysis_enemy_5/SA_results.txt", delim_whitespace=True)
    data2 = pd.read_csv("FINALSensitivityAnalysis_enemy_7/SA_results.txt", delim_whitespace=True)
    data3 = pd.read_csv("FINALSensitivityAnalysis_enemy_8/SA_results.txt", delim_whitespace=True)

    plot_SA_results(data1, data2, data3)
