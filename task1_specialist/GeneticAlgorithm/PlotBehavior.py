import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind

# Read data into a DataFrame
df = pd.read_csv('FINALBehavioral_evaluation.csv', header=0)

# Define the metrics for which you want to create boxplots
metrics = ["player_health", "enemy_health", "gain", "time", "lefts", "rights", "jumps", "releases", "shots", "accuracy"]

# Create a 9x2 grid of boxplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(7, 10))

# Define the colors for each 'config' group
colors = ['#4272f5', '#f5425a']

# Dictionary to store t-test results
t_test_results = {}

for i, metric in enumerate(metrics):
    # Group the data by 'config'
    grouped_data = [df[df['config'] == config][metric] for config in df['config'].unique()]
    
    # Create boxplots with custom colors
    box = axes[i // 2, i % 2].boxplot(grouped_data, patch_artist=True, showfliers=True, labels=df['config'].unique())

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)  # Box color
    
    for whisker in box['whiskers']:
        whisker.set(color='gray', linewidth=0.5)  # Whisker color
    
    for cap in box['caps']:
        cap.set(color='gray', linewidth=0.5)  # Whisker cap color
    
    for median in box['medians']:
        median.set(color='black', linewidth=1.5)  # Median line color
    
    # Calculate t-test for each pair of 'config' groups
    configs = df['config'].unique()
    for j in range(len(configs)):
        for k in range(j+1, len(configs)):
            config1 = configs[j]
            config2 = configs[k]
            data1 = df[df['config'] == config1][metric]
            data2 = df[df['config'] == config2][metric]
            t_stat, p_value = ttest_ind(data1, data2)
            t_test_results[(config1, config2)] = (t_stat, p_value)

    # Include t-test results in the title
    title = f"{metric}\n"
    for (config1, config2), (t_stat, p_value) in t_test_results.items():
        title += f"(T-test: t-stat={t_stat:.2f}, p-value={p_value:.2f})"
    
    axes[i // 2, i % 2].set_title(title, fontsize=11)

# Adjust layout
plt.suptitle('Behavioral Evaluation')
plt.tight_layout()
# plt.show()
plt.savefig('FINALBehavioral_evaluation.png')
